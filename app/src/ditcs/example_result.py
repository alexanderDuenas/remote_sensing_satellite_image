from app.src.utils.example import format_date


class ExampleResult:

    def __init__(self, date, counter_limit, value, second_value, message):
        self.date = date
        self.counter_limit = counter_limit
        self.value = value
        self.second_value = second_value
        self.message = message

    def get_result(self):
        return {
            "date": format_date(self.date),
            "counter_limit": self.counter_limit,
            "value": self.value,
            "second_value": self.second_value,
            "message": self.message
        }
