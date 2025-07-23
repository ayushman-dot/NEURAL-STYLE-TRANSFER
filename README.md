# NEURAL-STYLE-TRANSFER

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: AYUSHMAN JAISWAL

*INTERN ID*: CT04DG3348

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION* : 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DISCREPTION*

Neural Style Transfer (NST) is a fascinating application of deep learning that allows us to create artwork by combining two images: a content image (typically a photograph) and a style image (such as a painting). The goal is to generate a new image that retains the content of the photograph but is painted in the artistic style of the reference artwork. This technique beautifully merges the worlds of computer vision and creativity, producing results that are not only technically impressive but visually striking.

At the heart of this system is a pre-trained VGG19 convolutional neural network, originally designed for image classification. In this project, we repurpose this model to extract meaningful features from both the content and style images. The style is captured by analyzing the correlations between feature maps using what’s known as a Gram matrix, while the content is retained through high-level feature comparison. By minimizing the difference between the content features of the target and content images, and simultaneously minimizing the difference between the style features of the target and style images, we gradually update the target image until it reflects both content and style in harmony.

This implementation uses PyTorch and runs efficiently on a modern CPU or GPU. The input images are preprocessed and normalized to fit the VGG model’s expectations. The content image is typically a photo—say, of a person or landscape—while the style image can be anything from a Van Gogh painting to a sketch or abstract pattern. The algorithm performs backpropagation, not to train a model, but to update the pixels of the target image directly. This process iteratively refines the image to become more like the style image in texture, brush strokes, and color, while preserving the original structure of the content image.

Running this model can feel like watching art being painted by a machine. At each iteration, the stylized image becomes clearer and more defined, blending the aesthetics of the chosen painting with the realism of the photo. What makes neural style transfer particularly exciting is that it's not limited to academic or research use. It has real-world applications in creative industries, content creation, advertising, and even user-facing mobile apps.

*OUTPUT*

<img width="389" height="411" alt="Image" src="https://github.com/user-attachments/assets/1bb1c77d-947c-4989-883a-f4f3991dd0f0" />
