# Проект манипулятора-сортировщика мусора с компьютерным зрением.

Манипулятор-сортировщик умеет определять местоположение объекта мусора в рабочей зоне и затем классифицировать и сортировать его.
Классифицируемые типы мусора: пластик, металл, картон, стекло, бумага. В проекте используется модель нейросети [MobileNet v3](https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa), предобученная на датасете ImageNet.

### Waste224x224.zip:
датасет с фото мусора.
### waste_sorter_model_train_and_test.ipynb:
ноутбук для обучения нейросети.
### mobilenetv3_large_100_waste.pth:
обученная на фото мусора модель нейросети.
### cardboard145.jpg, glass25.jpg ...
тестовые изображения.
