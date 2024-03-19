# Определение авторства текстов

## Задача

Создать ML-сервис для определения авторства текстов на основании фрагмента из
5-10 предложений.

## Состав команды

- [Дарья Мишина](https://github.com/DariaMishina)
- [Кирилл Рубашевский](https://github.com/kirill-rubashevskiy)
- [Дмитрий Шильцов](https://github.com/DmitriyShiltsov)
- [Елена Вольф](https://github.com/Graf-D) (куратор)

## Работа с сервисом

Сервис развернут на render.com и состоит из:

- [FastAPI приложения с обученной ML-моделью](https://mlds23-authorship-identification-app.onrender.com)
  ([документация](https://mlds23-authorship-identification-app.onrender.com/docs)
  по работе с ним) и
- [Telegram бота](https://t.me/mlds23_ai_bot) (подсказку по его работе можно
  получить в стартовом меню)

## Инструкция для локального запуска сервиса

<details>

<summary>для запуска FastAPI приложения</summary>

1. Клонировать репозитарий:

```bash
git clone https://github.com/kirill-rubashevskiy/mlds23-authorship-identification.git
```

2. Установить [Docker](https://docs.docker.com/get-docker/),
   [PostgreSQL](https://www.postgresql.org/download/) и
   [Redis](https://redis.io/docs/install/install-redis/)
3. Поднять PostgreSQL и создать базу данных и пользователя с паролем для работы
   с ней (можно использовать свои название базы данных, имя пользователя и
   пароль):

```postgresql
CREATE DATABASE db_name;
CREATE USER db_user WITH PASSWORD 'db_password';
GRANT ALL PRIVILEGES ON DATABASE db_name TO db_user;
```

4. Создавать Docker-образ приложения, создать и запустить Docker-контейнер
   (команда запускается в корневой папке проекта):

```bash
# значения переменных DB_NAME, DB_PASSWORD и DB_USER берем из шага 2
# при изменении порта APP_PORT необходимо также поменять «левый» порт в port mapping (флаг -p)
docker build -f app/Dockerfile -t app . && \
docker run \
--env APP_PORT=8000 \
--env DB_HOST=host.docker.internal \
--env DB_NAME=db_name \
--env DB_PASSWORD=db_password \
--env DB_PORT=5432 \
--env DB_USER=db_user \
--env REDIS_HOST=host.docker.internal \
--name app \
-p 8000:8000 \
app
```

FastAPI приложение будет доступно по адресу: http://0.0.0.0:8000

</details>

<details>

<summary>для запуска FastAPI приложения вместе с Telegram ботом </summary>

1. Клонировать репозитарий:

```bash
git clone https://github.com/kirill-rubashevskiy/mlds23-authorship-identification.git
```

2. Установить [Docker](https://docs.docker.com/get-docker/),
   [PostgreSQL](https://www.postgresql.org/download/),
   [Redis](https://redis.io/docs/install/install-redis/) и
   [ngrok](https://ngrok.com/download)

3. Поднять PostgreSQL и создать базу данных и пользователя с паролем для работы
   с ней (можно использовать свои название базы данных, имя пользователя и
   пароль):

```postgresql
CREATE DATABASE db_name;
CREATE USER db_user WITH PASSWORD 'db_password';
GRANT ALL PRIVILEGES ON DATABASE db_name TO db_user;
```

4. Создать бота в Telegram при помощи [BotFather](https://telegram.me/BotFather)
   (при создании будет сгенерирован токен)

5. Создать HTTPS endpoint для бота при помощи ngrok:

```bash
# порт для бота можно выбрать по своему усмотрению
ngrok http 8080 # порт для бота может быть любым
```

6. создать в корневой папке репозитария файл `.env` со следующими переменными
   окружения:

```dosini
# пример .env
APP_PORT=8000 # порт для FastAPI приложения, можно выбрать по своему усмотрению
DB_USER=db_user # имя пользователя из шага 3
DB_PASSWORD=db_password # пароль пользователя из шага 3
DB_HOST=host.docker.internal
DB_PORT=5432
DB_NAME=db_name # название базы данных из шага 3
REDIS_HOST=host.docker.internal
TOKEN=token # заменить на токен из шага 4
BASE_WEBHOOK_URL=base_webhook_url # заменить на forwarding URL из лога терминала после выполнения шага 5
APP_URL=http://app:8000/ # порт для FastAPI приложения
BOT_PORT=8080 # порт из шага 5
```

7. Создавать Docker-образы приложения и бота, создать и запустить
   Docker-контейнеры (команда запускается в корневой папке проекта):

```bash
docker compose up -d
```

FastAPI приложение будет доступно по адресу: http://0.0.0.0:8000 \
Бот будет доступен в Telegram по имени, выбранному на шаге 4

</details>

## Работа над проектом

### Предварительный план работы

1. Сбор данных (до 31 октября 2023 года)
2. Разведочный анализ данных (до 15 ноября 2023 года)
3. ML-эксперименты и создание MVP (до 4 марта 2024 года)
4. DL-эксперименты (срок TBA)
5. Доработка MVP в полноценный сервис и deployment (срок TBA)

### Сбор данных

На данном этапе мы:

- отобрали авторов для обучения моделей
- собрали тексты авторов из открытых источников при помощи разработанного
  парсера
- разместили собранные данные в S3 хранилище

<details>
  <summary><b>Подробнее</b></summary>

Для проекта были отобраны 10 классических русских писателей, живших в 19 веке:

- И. А. Бунин
- В. М. Гаршин
- Н. В. Гоголь
- А. И. Куприн
- Ф. М. Достоевский
- Д. Н. Мамин-Сибиряк
- А. П. Платонов
- А. С. Пушкин
- И. С. Тургенев
- А. П. Чехов

По каждому автору было собрано не менее 10 прозаических произведений.

Данные были собраны на
[сайте интернет-библиотеки Алексея Комарова](https://ilibrary.ru/)
([ноутбук](notebooks/01_parsing.ipynb) с примером работы парсера).

Собранные данные (тексты в формате .txt, а также .csv файл с общей информацией о
текстах) размещены в облачном хранилище Yandex Object Storage. На
[сайте хранилища](https://mlds23-authorship-identification.website.yandexcloud.net)
доступна таблица с полной информацией о собранных текстах.

</details>

### Разведочный анализ данных

На данном этапе мы:

- проанализировали статистики из текстов
- критически оценили нграммы (уни-, би- и триграммы)
- проанализировали части речи (в целом и для предложений разной длинны)
- провели тематическое моделирование

<details>
  <summary><b>Подробнее</b></summary>

#### Статистики из текстов

([ноутбук](notebooks/05_eda_dm_stats_ngrams.ipynb) с анализом статистик из
текстов и нграмм, [ноутбук](notebooks/05_eda_shiltsov_punct_grams.ipynb) с
анализом статистик по пунктуации и частям речи в зависимости от длины
предложений)

Для начала мы посчитали среднюю длину слова и предложения по каждому автору:

![mean_word_length](visualizations/mean_word_length.png)
![mean_sentence_length](visualizations/mean_sentence_length.png)

На графиках видно, длина слов и предложений у авторов различаются, поэтому было
принято решение сделать углубленный анализ следующих статистик с помощью пакета
`ruts`:

- количество предложений
- количество слов
- количество уникальных слов
- количество длинных слов
- количество сложных слов
- количество простых слов
- количество односложных слов
- количество многосложных слов
- количество символов
- количество букв
- количество пробелов
- количество слогов
- количество знаков препинания

В результате датасет `splitted_df` с уже расспличенными по 550 токенов
произведениями был дополнен колонками, перечисленными выше, и сохранен в s3 как
`ruts_df`.

Также был сделан анализ использования знаков пунктуации у разных авторов в
предложениях различной длины:

**Вывод: авторы по-разному строят предложения разной длины с точки зрения знаков
препинания, возможно это может дать полезные дополнительные признаки для
обучения моделей.**

![punct_distribution](visualizations/punct_distribution.png)

На следующем этапе мы попробуем использовать эти статистики в качестве признаков
для классификации.

#### Нграммы

([ноутбук](./EDA/EDA_DM.ipynb) с анализом статистик из текстов и нграмм)

Для расширения стоплиста слов мы проанализировали самые часто встречающиеся
слова:

![top_20_words](visualizations/top_20_words.png)

На графике видны погрешности в предобработке (аномально длинное тире с
пробелами); 50 самых часто встречающихся слов сохранены в s3 для последующего
анализа как `count_words`.

Мы также проанализировали топ10 биграмм и триграмм:

![top_10_bigrams](visualizations/top_10_bigrams.png)
![top_10_trigrams](visualizations/top_10_trigrams.png)

#### Анализ частей речи

([ноутбук](notebooks/05_eda_kr_pos_topic_modeling.ipynb) с анализом частей речи
и тематическим моделированием,
[ноутбук](notebooks/05_eda_shiltsov_punct_grams.ipynb) с анализом статистик по
пунктуации и частям речи в зависимости от длины предложений)

Для анализа частей речи мы:

- просемплировали объекты таким образом, чтобы по каждому автору было одинаковое
  количество объектов
- провели предобработку данных:
  - привели тексты к нижнему регистру
  - удалили http ссылки, цифры и пунктуацию
  - токенизировали тексты
  - определили часть речи каждого токена при помощи библиотеки `spaCy`
- по каждому объекту посчитали количество вхождений каждой части речи и
  нормализовали данные (чтобы сумма вхождений всех частей речи в объект давала
  единицу)

Затем мы проанализировали распределения частей речи по всем объектам, и
статистики распределений по автору.

Распределения значений частей речи соответствуют одному из четырех типов:

![pos_distribution_types](visualizations/pos_distribution_types.png)

Мы проверили, что мультимодальное распределение отдельных частей речи
наблюдается у всех авторов (гипотеза подтвердилась), и для дальнейшего анализа
оставили только части речи с нормальным и скошенным нормальным распределениями.

Мы анализировали медиану, которая меньше подвержена выбросам:

![pos_medians_by_author](visualizations/pos_medians_by_author.png)

Авторы различаются по медианной частоте использования частей речи:

- Куприн использует больше нарицательных и прилагательных, но меньше глаголов и
  местоимений
- Пушкин использует меньше наречий и сочинительных союзов
- Бунин использует больше нарицательных, прилагательных и предлогов, но меньше
  местоимений и глаголов и т.д.

При этом понижение размерности данных при помощи PCA (мы дополнительно
ограничили значения сверху 99-м персентилем и стандартизировали данные) и TSNE
не позволило кластеризовать авторов:

![pos_tsne_pca](visualizations/pos_tsne_pca.png)

**Вывод: статистики по частям речи могут быть дополнительными признаками для
обучения моделей, но эффект от них (без нелинейного преобразования) будет
ограничен при использовании линейных моделей.**

Мы разместили сгенерированные признаки (нормированное количество каждой части
речи в объекте) в облачном хранилище Yandex Object Storage по проекту для их
использования при обучении моделей.

Мы также проанализировали распределение частей речи в зависимости от длины
предлоожений:

![pos_distribution_by_length](visualizations/pos_distribution_by_length.png)

**Вывод: авторы по-разному строят предложения разной длины с точки зрения
употребления частей речи, возможно это может дать полезные дополнительные
признаки для обучения моделей.**

#### Тематическое моделирование

([ноутбук](notebooks/05_eda_kr_pos_topic_modeling.ipynb) с анализом частей речи
и тематическим моделированием)

Для тематического моделирования мы использовали все объекты датасета и три части
речи: существительные (нарицательные), прилагательные и глаголы.

Перед тематическим моделированием мы провели предобработку текстов:

- привели тексты к нижнему регистру,
- удалили http ссылки, цифры, пунктуацию
- удалили стоп-слова: неправильно классифицированные имена собственные
  (`аркадий`, `наташа`и т.п.) и слова, которые не несут смысловой нагрузки
  (`хотеть`, `быть` и т.п.)
- удалили слишком редкие (встречающиеся менее 3 раз) и слишком частые
  (встречающихся более, чем в половине объектов) слова
- токенизировали или лемматизировали тексты

Для тематического моделирования мы использовали алгоритм LDA. Мы создали
несколько моделей в зависимости от количества предсказываемых ими тем (от 10 до
30 с шагом 5), и для оценки качества работы моделей использовали оценку асессора
(меня) на 10 случайных объектах.

Каждая модель смогла (субъективно) корректно определить темы от 0 до 3 объектов:

- модель с 10 темами — природу (объект 1), выражение эмоций (объект 2) и лошадей
  (объект 5)
- модель с 20 темами — происходящее на улице (объект 5) и внешние характеристики
  героя (объект 10)
- модель с 25 темами — происходящее на улице (объект 5)
- модель с 30 темами — природу (объект 1), помещения (объект 3) и внешние
  характеристики героя (объект 10)

Определить темы отдельных объектов (например, 4 и 7) не смогли как модели, так и
асессор; возможные причины:

- деление на объекты не учитывает абзацы, которые отдельные авторы используют
  для разграничения тем
- с учетом ограничения на максимальное количество слов в объекте, отбора частей
  речи и удаления стоп-слов после предобработки в объектах не остается
  достаточное количество несущих смысл слов для корректного определения темы

**Вывод: использование предсказанных моделями тем и их вероятностей
нецелесообразно использовать в качестве признаков для обучения моделей,
классифицирующих авторов.**

</details>

### ML-эксперименты и создание MVP

На данном этапе мы:

- провели эксперименты с различными предобработкой данных (удаление стоп-слов,
  удаление пунктуации, лемматизация), способами создания признаков
  (Bag-of-Words, TF-IDF, текстовые статистики) и семействами моделей (линейные
  модели, вероятностные модели, KNN, решающие деревья и их ансамбли)
- разработали MVP в формате ТГ-бота

MVP развернут на render.com и доступен по ссылке https://t.me/mlds23_ai_bot

### DL-эксперименты

На данном этапе планируется:

- использовать трансформеры (с дообучением на собранном датасете)
- дообучить LLM-модель (Llama, Mistral, Saiga)
- обучить классические ML-модели на эмбеддингах из DL-модели

### Доработка MVP в полноценный сервис и deployment

На данном этапе планируется доработать MVP в полноценный веб сервис
