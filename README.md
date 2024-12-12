# 🧠 This Is The ML Path For Capstone Project "Urkins" 🧠
## Models
There are two models for our mobile applications: Skin Conditions Model and Skin Type Model.

### Skin Type Model
Skin type model uses RestNet50 for transfer learning because its good at recognising texture, so for our apllication to detect user skin type our model model has to be able to clasify the user skin texture accurately, wether it is dry, oily, or normal.

### Skin Conditions Model
Skin condition model uses MobileNetV2 for transfer learning because its good at recignising shape and pattern, so for our application to detect user ksin conditions our model has to able to clasify the user skin pattern, like acne and shape like eye bags.

## Tester
Other than the models, we also have some tester to test our models using streamlit to check if there are bugs or error with the models.