from aiogram.fsm.state import State, StatesGroup


class PredictText(StatesGroup):
    entering_text = State()


class PredictTexts(StatesGroup):
    uploading_file = State()


class Rate(StatesGroup):
    entering_rating = State()
