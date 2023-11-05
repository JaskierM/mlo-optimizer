class CoefficientDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_coefficient(value)
        setattr(instance, self.name, value)

    def verify_coefficient(self, value):
        if type(value) not in (int, float):
            raise TypeError(f'Valid types for attribute "{self.name[2:]}" are int and float')
        if value <= 0:
            raise TypeError(f'Attribute "{self.name[2:]}" must be greater than 0')
