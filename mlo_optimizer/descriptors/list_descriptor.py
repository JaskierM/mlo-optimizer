class ListDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.verify_list(value)
        setattr(instance, self.name, value)

    def verify_list(self, value):
        if isinstance(value, list):
            raise TypeError(f'Attribute "{self.name[2:]}" must be represented by a Python list')
        if not value:
            raise TypeError(f'Attribute "{self.name[2:]}" must not be empty')
