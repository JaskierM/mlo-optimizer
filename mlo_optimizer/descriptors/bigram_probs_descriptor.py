class BigramProbsDescriptor:
    def __set_name__(self, owner, name):
        self.name = '__x'

    def __get__(self, instance, owner):
        return getattr(instance, self.name)
