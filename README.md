Суть проекта заключается в том, что нужно сделать нейронную сеть, которая будет распознавать лица на изображении. И Есть следующие условия:  
Модель должна работать быстро (предполагается запуск на Raspberry py, нужна стабильная работа хотя бы больше 5 кадров в секунду (а лучше намного больше). Также должна мало весить  
Есть всего два класса — background и лицо. Модель должна детектировать лица с достаточной точностью (вблизи должна распознавать их околоидеально)  
Т.к. распознаваться на изображении будут лица, то применять нужно модель, которая распознаёт именно маленькие объекты  
По размерам обнаруженных объектов должен находиться самый маленький из них  
Программа должна рассчитывать расстояние от центра изображения до каждого из объектов  
Потенциально нужно будет сделать трекинг

Соответственно, следующий план:  
Создать докер-файл (для использования докер-контейнера), чтобы в нём тестировать приложение  
Определиться с разрешением фото, которые будут использоваться  
Подобрать датасет для transfer-learning, подобрать инструмент для разметки этого датасета (для детектирования)  
Подобрать модель, соответствующую выдвинутым критериям  
Дообучить модель, протестировать её на своей тестовой части  
Добавить расчёт расстояния от центра изображения до каждого объекта  
Добавить сортировку по величине  
Добавить какую-нибудь подсветку, когда центр изображения попадает внутрь рамки объекта (не обязательно в центр объекта)  

В будущем:  
Добавить трекинг  
Добавить Raspberry py, купить к нему камеру, поставить нейронку на микроконтроллер
