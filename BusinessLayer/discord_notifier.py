"""Gửi cảnh báo "có người đến" lên Discord qua webhook.

  * Lắng nghe EventBus (PERSON_KNOWN / PERSON_UNKNOWN), KHÔNG chặn luồng AI: callback chỉ xếp sự kiện vào hàng đợi.
  * Một luồng riêng gửi HTTP. Giữa 2 tin tối thiểu `min_interval` giây (mặc định 20s).
  * Chờ bằng Condition.wait(timeout): không sleep, không vòng lặp bận -> 0% CPU khi rảnh,
    và thức dậy ngay khi có sự kiện mới hoặc khi tắt app.
  * Sự kiện đến trong lúc đang chờ không bị bỏ: được gộp thành 1 tin (mỗi sự kiện 1 dòng) gửi ngay khi hết thời gian chờ.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from Utils.event_bus import PERSON_KNOWN, PERSON_UNKNOWN, EventBus
from Utils.logger import get_logger

log = get_logger("Discord")

MAX_LINES_PER_MSG = 15      # 15 dòng * ~60 ký tự << giới hạn 2000 ký tự của Discord
MAX_PENDING = 200           # chặn hàng đợi phình to nếu Discord/mạng lỗi kéo dài


class DiscordNotifier:
    def __init__(self, webhooks: list[str], min_interval: float = 20.0, timeout: float = 10.0,
                 username: str = "Smart Vision"):
        self.webhooks = list(webhooks)
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = float(timeout)
        self.username = username

        self._cv = threading.Condition()
        self._pending: list[tuple[float, str, str | None]] = []   # (ts, topic, tên)
        self._next_allowed = 0.0                                  # mốc time.monotonic()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="DiscordNotifier", daemon=True)

    # ------------------------------------------------------------------ khởi tạo
    @classmethod
    def from_config(cls, cfg, bus: EventBus) -> "DiscordNotifier | None":
        """Đọc discord.web_hooks (config.yaml + config.local.yaml). Trả về None nếu chưa cấu hình."""
        dc = cfg.section("discord")
        hooks = dc.get("web_hooks") or []
        if isinstance(hooks, str):
            hooks = [hooks]
        valid = []
        for h in hooks:
            h = str(h).strip()
            if not h:
                continue
            if not h.startswith("https://"):      # chặn file://, http:// ...
                log.warning("Bỏ qua webhook không hợp lệ (phải bắt đầu bằng https://)")
                continue
            valid.append(h)
        if not dc.get("enabled", True) or not valid:
            log.info("Discord: tắt hoặc chưa có discord.web_hooks -> không gửi cảnh báo.")
            return None
        notifier = cls(valid, min_interval=float(dc.get("min_interval_sec", 20)))
        notifier.attach(bus)
        notifier.start()
        log.info("Discord: bật cảnh báo (%d webhook, cách nhau >= %.0fs).", len(valid), notifier.min_interval)
        return notifier

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(PERSON_KNOWN, self.on_event)
        bus.subscribe(PERSON_UNKNOWN, self.on_event)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ nhận sự kiện (chạy ở luồng AI, phải nhanh)
    def on_event(self, ev: dict) -> None:
        with self._cv:
            self._pending.append((float(ev.get("ts", time.time())), ev.get("topic", ""), ev.get("name")))
            if len(self._pending) > MAX_PENDING:
                del self._pending[0]
            self._cv.notify()

    # ------------------------------------------------------------------ luồng gửi
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stopped:
                    if not self._pending:
                        self._cv.wait()                                 # rảnh: ngủ hẳn tới khi có sự kiện
                        continue
                    wait = self._next_allowed - time.monotonic()
                    if wait <= 0:
                        break                                           # đã hết cooldown -> gửi
                    self._cv.wait(timeout=wait)                         # chờ đúng phần còn lại của cooldown
                if self._stopped:
                    return
                batch = self._pending[:MAX_LINES_PER_MSG]
                del self._pending[:len(batch)]

            retry_after = self._send("\n".join(self._format(*b) for b in batch))

            with self._cv:
                self._next_allowed = time.monotonic() + max(self.min_interval, retry_after)

    @staticmethod
    def _format(ts: float, topic: str, name: str | None) -> str:
        when = time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(ts))
        if topic == PERSON_KNOWN and name:
            return f"🟢 **{name}** — {when}"
        return f"🔴 **Người lạ** — {when}"

    # ------------------------------------------------------------------ HTTP
    def _send(self, content: str) -> float:
        """Gửi tới mọi webhook. Trả về số giây cần chờ thêm nếu Discord báo 429 (0 nếu ổn)."""
        body = json.dumps({
            "username": self.username,
            "content": content,
            "allowed_mentions": {"parse": []},      # tên người dùng nhập không thể kích hoạt @everyone
        }).encode("utf-8")
        retry_after = 0.0
        for i, url in enumerate(self.webhooks, 1):
            req = urllib.request.Request(url, data=body, method="POST", headers={
                "Content-Type": "application/json",
                "User-Agent": "SmartVision/1.0",    # Discord/Cloudflare chặn User-Agent mặc định của urllib
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout):
                    pass
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = self._parse_retry_after(e)
                    retry_after = max(retry_after, wait)
                    log.warning("Webhook #%d bị giới hạn tốc độ, thử lại sau %.0fs", i, wait)
                else:
                    log.warning("Webhook #%d lỗi HTTP %s", i, e.code)    # không log URL vì chứa token
            except Exception as e:  # noqa: BLE001
                log.warning("Webhook #%d gửi thất bại: %s", i, type(e).__name__)
        return retry_after

    @staticmethod
    def _parse_retry_after(e: urllib.error.HTTPError) -> float:
        try:
            return float(e.headers.get("Retry-After") or json.loads(e.read().decode()).get("retry_after", 5))
        except Exception:  # noqa: BLE001
            return 5.0