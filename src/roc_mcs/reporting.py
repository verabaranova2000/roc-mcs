def format_run_report(output_folder, files_created, diagnostics=None):
    lines = []
    lines.append("=" * 50)
    lines.append("Обработка завершена")
    lines.append("")
    lines.append("Каталог результатов:")
    lines.append(f"  {output_folder}")
    lines.append("")
    lines.append("Сформированные файлы:")

    for f in files_created:
        lines.append(f"  • {f}")

    if diagnostics:
        lines.append("")
        lines.append("Диагностические графики:")
        for d in diagnostics:
            lines.append(f"  • qc/{d}.png")

    lines.append("=" * 50)
    return "\n".join(lines)



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