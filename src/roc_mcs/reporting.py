from pathlib import Path

def print_run_report(output_folder, files):
    output_folder = Path(output_folder)

    print("=" * 50)
    print("Обработка завершена")
    print()
    print("Каталог результатов:")
    print(f"  {output_folder.resolve()}")
    print()
    print("Сформированные файлы:")

    for file in files:
        file = Path(file)
        try:
            rel = file.relative_to(output_folder)
        except ValueError:
            rel = file.name
        print(f"  • {rel}")

    print("=" * 50)