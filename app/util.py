from torch import nn
from torchvision import models
import lime.lime_image
from skimage.segmentation import mark_boundaries
import torch

def get_model():
    model = models.densenet121(pretrained=False)  # No pretrained weights
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 5)  # Modify for 5 classes

    # Load weights
    weights_path = "densenet121_fold5.pth"
    try:
        # Load the saved file
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

        # If checkpoint is a full state dict with extra info, extract the model state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'module.' prefix if model was saved with DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        # Load state dict into model
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise

    # Move to device and set to evaluation mode
    model.eval()
    return model


explainer = lime.lime_image.LimeImageExplainer()


def batch_predict(images):
    model = get_model()
    model.eval()
    batch = torch.tensor(images).permute(0, 3, 1, 2).float()  # Convert to PyTorch format
    outputs = model(batch)
    return torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()


# Get model prediction for the selected image
def get_lime(test_image):
    print(test_image.shape)
    image_np = test_image[0].permute(1, 2, 0).numpy()
    model = get_model()
    model.eval()
    output = model(test_image)
    predicted_label_idx = torch.argmax(output, dim=1).item()
    label_mapping = {0: 'ASC_H', 1: 'ASC_US', 2: 'HSIL', 3: 'LSIL', 4: 'NILM'}
    predicted_label = label_mapping[predicted_label_idx]
    # Convert numeric label to class name
    print(predicted_label)

    # Generate explanation using LIME
    explanation = explainer.explain_instance(
        image_np,
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=10  # Number of perturbations
    )

    # Get the most relevant region from LIME explanation
    temp, mask = explanation.get_image_and_mask(
        predicted_label_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # LIME Explanation Image
    highlighted_image = mark_boundaries(temp, mask)
    return highlighted_image





