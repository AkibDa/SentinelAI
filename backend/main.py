# backend/main.py

import uvicorn

if __name__ == "__main__":
    # Note the change to "app.main:app"
    uvicorn.run("app.app:app", host="0.0.0.0", port=8080, reload=True)