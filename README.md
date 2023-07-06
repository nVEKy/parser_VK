# parser_VK
<p>Разделим весь проект на три части:</p>
<ol>
  <li>Парсер</li>
  <li>Фильтр</li>
  <li>Датасет</li>
</ol>
<h1 align="center">Парсер</a></h1>
<p>Итак, для получения записей и комментариев к ним используется <a href="https://dev.vk.com/method" target="_blank">VK API</a></p>
<p>А именно следующие модули:</p>
<ul>
  <li><a href="https://dev.vk.com/method/groups.search" target="_blank">groups.search</a></li>
  <li><a href="https://dev.vk.com/method/wall.get" target="_blank">wall.get</a></li>
  <li><a href="https://dev.vk.com/method/wall.getComments" target="_blank">wall.getComments</a></li>
</ul>
<p>Каждый из этих методов возвращает объект типа json, из которого достаётся вся необходимая информация. Последняя сохраняется в файлы posts_parsed.csv (для постов), comments_parsed.csv (для комментариев)</p>
<h1 align="center">Фильтр</a></h1>
<p>Из полученных в результате работы парсера файлов достаётся вся информация и начинается фильтрация.</p>
<p>Информация фильтруется по следующим критериям:</p>
<ol>
  <li>Для постов:</li>
  <ul>
    <li>длина текста > 10 слов</li>
    <li>соответствие теме психологии</li>
    <li>наличие хотя бы 1 именованной сущности</li>
    <li>отсутствие грамматических ошибок</li>
    <li>отсутствие спама и рекламы в тексте публикации</li>
    <li>количество комментариев под публикацией > 1 комментария</li>
  </ul>
  <li>Для комментариев:</li>
  <ul>
    <li>длина текста >= 4 слов</li>
    <li>соответствие теме психологии</li>
    <li>наличие хотя бы 1 именованной сущности из публикации</li>
    <li>отсутствие грамматических ошибок</li>
    <li>отсутствие спама и рекламы в тексте комментария</li>
  </ul>
</ol>
<p>Критерии соответствия теме психологии и отсутствия спама и рекламы в тексте проверяется моделью, обученной на сформированных заранее .csv файлах "clean_text_psy.csv" и "clean_text_ads.csv", соответсвенно. Об этом в части Датасет.</p>
<p>Отсутствие грамматических ошибок не проверяется из-за некоторых сложностей, возникших в ходе реализации (возможно, будет реализовано позже).</p>
<p>Проверка наличия хотя бы 1 именованной сущности из публикации для комментариев заключается в простом поиске поста, к которому относится комментарий, и сравнение их множест, именнованные сущности переведены в начальные формы для простоты сравнения/p>
<h1 align="center">Датасет</h1>
<p>Были собраны три датасета:</p>
<ol>
  <li>Тексты на тему психологии</li>
  <li>Тексты на любую тему, кроме психологии</li>
  <li>Тексты с рекламой</li>
</ol>
<p>Каждый датасет хранится в папке "texts/...", в зависимости от его содержания.</p>
<p>Программа get_texts.py формирует из этих датасетов два файла, которые помещаются в папку class: "clean_text_psy.csv" и "clean_text_ads.csv". Также формируются промежуточные фалы (они просто есть).</p>
<p>Два наших основных файла формируются следующим образом: убираются все знаки пунктуации, а также стоп-слова, такие как "и", "что" и др.</p>
<p>На основе данных уже не текстах, но наборах слов, обучается модель по алгоритму стохастического градиентного спуска (SGD). Как сохранить такую модель я найти не смог (возможно, плохо пытался), поэтому это происходит непременно до фильтрации.</p>
<p>Данное решение зято из поста на хабре: <a href=https://habr.com/ru/articles/538458/>Решаем natural language processing-задачу – классификация текстов по темам</a>.</p>
