from enum import Enum


class StrEnum(Enum):
    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")

    @classmethod
    def get_description(cls, value):
        return cls._missing_(value).description


# utils.py
REQUIRED_FAME = ['C17', 'C18', 'C18:1 I', 'C18:1 II', 'C18:2', 'C18:3', 'C20:1']
REQUIRED_EPO = [
    'C18:1 EPO', 'C18:2 1-EPO I', 'C18:2 1-EPO II', 'C18:3 1-EPO I',
    'C18:3 1-EPO II', 'C18:3 1-EPO III', 'C20:1 1-EPO',
    'C18:2 2-EPO', 'C18:3 2-EPO I'
]
