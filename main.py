import yaml
from src.train import train_model
from src.inference import predict

if __name__ == "__main__":
    # Charger la config
    with open("configs/resnet18.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Lancer l’entraînement
    train_model(
        data_dir=config["data"]["train_dir"],
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"],
        save_path="models/resnet18_waste.pth"
    )

    # Exemple d’inférence
    classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    print("Prediction:", predict("./tests/img/cardboard.jpeg", "./models/resnet18_waste_1.pth", classes))