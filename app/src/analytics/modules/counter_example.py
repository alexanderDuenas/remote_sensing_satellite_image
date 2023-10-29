from time import sleep


def counter_example(params):
    print("Algoritmo iniciado.")
    counter = 1
    while counter <= params:
        print(f"Exemplo de algoritmo em execução: {counter}")
        counter += 1
        sleep(1)
    else:
        print("Algoritmo finalizado.")
