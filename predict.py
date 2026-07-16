
"""
predict.py — standalone inference for Xception + Quantum brain tumor model
Usage:
    python predict.py --image path/to/brain_mri.jpg
"""
import argparse, json, numpy as np, cv2, tensorflow as tf
from quantum_layer import QuantumLayer

def load_model(model_path="xcept_quant_model.keras"):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"QuantumLayer": QuantumLayer}
    )

def preprocess(img_path, img_size=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0        # ✅ ONLY this
    # ❌ DELETE the xception.preprocess_input line
    return np.expand_dims(img, 0)

def predict(img_path, model, classes):
    x    = preprocess(img_path)
    prob = model.predict(x, verbose=0)[0]
    idx  = int(np.argmax(prob))
    return {
        "prediction":  classes[idx],
        "confidence":  float(prob[idx]),
        "all_probs":   {c: float(p) for c, p in zip(classes, prob)},
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--model",  default="xcept_quant_model.keras")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open("classes.json") as f:
        classes = json.load(f)

    model  = load_model(args.model)
    result = predict(args.image, model, classes)

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.1%}")
    print("All probs  :", result["all_probs"])
