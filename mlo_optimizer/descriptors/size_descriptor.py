class SizeDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_size(value)
        setattr(instance, self.name, value)

    def verify_size(self, value):
        if isinstance(value, int):
            raise TypeError(f'Valid type for attribute "{self.name[2:]}" is int')
        if value <= 0:
            raise TypeError(f'Attribute "{self.name[2:]}" must be greater than 0')
