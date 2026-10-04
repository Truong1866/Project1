import uvicorn

if __name__ == "__main__":
    print("Khởi động Web Server tại http://localhost:8000")
    uvicorn.run("PresentLayer.present_api:app", host="0.0.0.0", port=8000, reload=False)