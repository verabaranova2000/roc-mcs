import os
import time

"""
Читатель CSV
------
Этот модуль решает главную задачу — он безопасно читает cycle_control_log_xxxxx.csv. Если строка еще не дописана в файл функцией add_csv, он просто возвращает указатель назад и ждет.
"""


class LiveCSVReader:
    def __init__(self, filepath, delimiter=';'):
        self.filepath = filepath
        self.delimiter = delimiter

    def follow(self):
        """Генератор, выдающий новые строки по мере их появления в файле."""
        # Ждем, пока файл физически не появится на диске
        while not os.path.exists(self.filepath):
            time.sleep(0.1)

        with open(self.filepath, 'r', encoding='utf-8-sig') as f:
            # 1. Ждем и читаем заголовки
            while True:
                pos = f.tell()
                line = f.readline()
                if line.endswith('\n'):
                    headers = line.strip().split(self.delimiter)
                    break
                f.seek(pos)
                time.sleep(0.05)

            # 2. Бесконечный цикл чтения новых данных
            while True:
                pos = f.tell()
                line = f.readline()
                
                # Если новых данных пока нет 
                if not line:
                    yield None  # нужно, чтобы генератор не просто молча спал, а сообщал графику: «Новых данных пока нет, но я работаю»
                    time.sleep(0.05)
                    continue
                    
                # Если строка полностью записана 
                if line.endswith('\n'):
                    values = line.strip().split(self.delimiter)
                    # Проверяем, что количество данных совпадает с заголовками
                    if len(values) == len(headers):
                        yield dict(zip(headers, values))
                else:
                    # Строка обрезана на середине (идет запись) — откатываемся и ждем 
                    f.seek(pos)
                    time.sleep(0.05)