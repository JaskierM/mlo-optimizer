class ProbabilityDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_probability(value)
        setattr(instance, self.name, value)

    def verify_probability(self, value):
        if type(value) not in (int, float):
            raise TypeError(f'Valid types for attribute "{self.name[2:]}" are int and float')
        if value < 0 or value > 1:
            raise TypeError(f'Probability "{self.name[2:]}" must be between 0 and 1')
