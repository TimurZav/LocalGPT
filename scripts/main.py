import uvicorn
import gradio as gr
from app import LocalGPT
from fastapi import FastAPI
from server.auth.auth_router import auth_router
from server.health.health_router import health_router


app = FastAPI()
app.include_router(health_router)
app.include_router(auth_router)
gr.mount_gradio_app(app, LocalGPT().launch_ui(), path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
