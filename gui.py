from tkinter import Tk, Button
from tkinter import font


class GUI:

    def __init__(self, board):
        self.app = Tk()
        self.app.title('TicTacToe')
        self.app.resizable(width=False, height=False)
        self.font = font.Font(family="Helvetica", size=32)
        self.buttons = {}
        for i in range(0, len(board)):
            x = int(i / 3)
            y = int(i % 3)
            button = Button(self.app, font=self.font, width=2, height=1)
            button.grid(row=y, column=x)
            self.buttons[x, y] = button
        button = Button(self.app, text='reset')
        button.grid(row=3, column=0, columnspan=3, sticky="WE")
        self.update(board, [None, None, None], False)

    def update(self, board, winning_combo, done):
        for i in range(0, len(board)):
            val = board[i]
            if val == 0:
                text = 'X'
            elif val == 1:
                text = '0'
            else:
                text = '.'
            x = int(i / 3)
            y = int(i % 3)
            self.buttons[x, y]['text'] = text
            self.buttons[x, y]['disabledforeground'] = 'black'
            if text == '.':
                self.buttons[x, y]['state'] = 'normal'
            else:
                self.buttons[x, y]['state'] = 'disabled'
        if done and winning_combo is None:
            for pos in winning_combo:
                x = int(pos / 3)
                y = pos % 3
                self.buttons[x, y]['disabledforeground'] = 'red'
            for x, y in self.buttons:
                self.buttons[x, y]['state'] = 'disabled'
        for i in range(0, len(board)):
            x = int(i / 3)
            y = int(i % 3)
            self.buttons[x, y].update()

    def mainloop(self):
        self.app.mainloop()
