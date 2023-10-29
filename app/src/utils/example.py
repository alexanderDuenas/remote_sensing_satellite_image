# TODO: Exemplo de função que pode ser utilizada por diversos métodos do projeto.

def format_date(date):
    if date is not None:
        return date.strftime("%d/%m/%Y %H:%M:%S")
