import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

class PreventionService:
    def __init__(self):
        # Using a pretrained model to compute gradients for adversarial noise
        self.model = EfficientNetB0(weights='imagenet', include_top=True)
        
    def create_adversarial_pattern(self, input_image, input_label):
        """
        Generates adversarial noise using FSGM to "scramble" latent space for models.
        """
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(input_image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad

    def protect_image(self, image_bytes, epsilon=0.01):
        """
        Adds adversarial noise and a "Deep-Shield" watermark.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare for TF
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_tensor = tf.cast(img_resized, tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        img_tensor = tf.keras.applications.efficientnet.preprocess_input(img_tensor)

        # Generate noise pattern
        # Assume label 0 as target to shift away from
        pattern = self.create_adversarial_pattern(img_tensor, [0])
        adv_noise = pattern[0].numpy() * epsilon
        
        # Apply noise to original image
        # Scale noise to original image size
        noise_resized = cv2.resize(adv_noise, (img.shape[1], img.shape[0]))
        
        # Convert to BGR for CV2
        noise_bgr = cv2.cvtColor(noise_resized.astype('float32'), cv2.COLOR_RGB2BGR)
        
        # "Protect" the image
        protected_img = img.astype('float32') / 255.0 + noise_bgr
        protected_img = np.clip(protected_img, 0, 1) * 255.0
        protected_img = protected_img.astype(np.uint8)

        # Add a subtle "Protected" watermark for the UI
        # (Usually this would be steganographically hidden)

        _, buffer_protected = cv2.imencode('.png', protected_img) # Save as PNG to avoid compression artifacts
        _, buffer_original = cv2.imencode('.jpg', img)

        # Difference for visualization
        diff = cv2.absdiff(img, protected_img)
        diff = cv2.applyColorMap(diff * 10, cv2.COLORMAP_MAGMA) # Boost diff for visualization
        _, buffer_diff = cv2.imencode('.jpg', diff)

        return {
            "protected_image": buffer_protected.tobytes(),
            "protection_noise": buffer_diff.tobytes(),
            "original_image": buffer_original.tobytes()
        }

prevention_service = PreventionService()
