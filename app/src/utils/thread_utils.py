import threading


def get_current_threads():
    current_threads = []
    for thread in threading.enumerate():
        current_threads.append(thread.name)
    return current_threads


def get_thread_if_exist(thread_name):
    for thread in threading.enumerate():
        if thread.name == str(thread_name):
            return thread
    return None
