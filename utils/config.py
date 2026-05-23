# utils/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    resnet18_waste: str
    classes: str
    resnet_waste_pth: str
    gradcam_chart: str
    data: str
    resnet_model: str
    resnet_onnx_model: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf8",
        extra="ignore"
    )

settings = Settings()