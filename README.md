# TenderHack
## Цель и задачи
Разработка прототипа механизма поиска товаров на портале поставщиков

В рамках хакатона участникам предстоит реализовать задачи: Проанализировать типовой функционал системы поиска товаров (поиск по ходу набора названия – подсказки, контекстные синонимы, транслитерация, опечатки, ранжирование подсказок и выдачи, поиск с учетом свойств, учет перестановки слов, история поиска, аналогичные товары и т.д.); Реализовать систему; Провести тестирование системы, оптимизировать ее работу в рамках улучшения метрик; Продемонстрировать работоспособность проекта; Определить возможности по масштабированию решения и следующим его доработкам.

![2022-10-31 23-27-55](https://user-images.githubusercontent.com/92402616/199105142-7a5a1e7a-f3b4-4166-8f26-07c21ee439c9.gif)


## Решение 

![Снимок1](https://user-images.githubusercontent.com/92402616/199107780-f95521fe-7034-4659-ab39-d1f29b6d490d.PNG)

![Снимо2к](https://user-images.githubusercontent.com/92402616/199108041-87c9bdc0-a43c-45f8-a270-a56cf1133111.PNG)



## Техническое описание проекта

- server.py - бэк на Flask, соединяет ML с фронтом
- search.py - постороение info search
- nlp.py - nlp обработка датасетов и запросов
- main.py - ml часть с языковой моделью Bert
- autocorrect.py - автокоррекция запросов
- autocomplete.py - автодополнение запросов в runtime
- template и static - фронт часть
- bert - обучение языковой модели



