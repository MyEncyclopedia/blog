import pygame


class GameBoard:
    PIECE_B = 'b'
    PIECE_W = 'w'

    def __init__(self, board_num=5, connect_num=3):
        self.grid_size = 10
        self.connect_num = connect_num
        self.start_x, self.start_y = 30, 50
        self.edge_size = self.grid_size / 2
        self.board_num = board_num
        self.piece = GameBoard.PIECE_B
        self.winner = None
        self.game_over = False
        self.action = None

        self.board = []
        for i in range(self.board_num):
            self.board.append(list("." * self.board_num))
        pygame.init()

        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("TTT")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.going = True


    def next_step(self):
        self.action = None
        while not self.action:
            self.update()
            self.render()
            self.clock.tick(60)
        return self.action

    def update(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.going = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self.handle_key_event(e)

    def next_user_input(self):
        self.action = None
        while not self.action:
            self.wait_user_input()
            self.render()
            self.clock.tick(60)
        return self.action

    def wait_user_input(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.going = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self.handle_user_input(e)

    def render(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.font.render("FPS: {0:.2F}".format(self.clock.get_fps()), True, (0, 0, 0)), (10, 10))

        self.draw(self.screen)
        if self.game_over:
            self.screen.blit(self.font.render("{0} Win".format("Black" if self.winner == 'b' else "White"), True, (0, 0, 0)), (500, 10))

        pygame.display.update()

    def available_actions(self):
        return [row * self.board_num + col for row in range(self.board_num) for col in range(self.board_num) if self.board[row][col] == '.']

    def handle_key_event(self, e):
        origin_x = self.start_x - self.edge_size
        origin_y = self.start_y - self.edge_size
        size = (self.board_num - 1) * self.grid_size + self.edge_size * 2
        pos = e.pos
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            if not self.game_over:
                x = pos[0] - origin_x
                y = pos[1] - origin_y
                r = int(y // self.grid_size)
                c = int(x // self.grid_size)
                if self.set_piece(r, c):
                    self.check_win(r, c)
                    self.action = (self.piece, r, c)

    def handle_user_input(self, e):
        origin_x = self.start_x - self.edge_size
        origin_y = self.start_y - self.edge_size
        size = (self.board_num - 1) * self.grid_size + self.edge_size * 2
        pos = e.pos
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            if not self.game_over:
                x = pos[0] - origin_x
                y = pos[1] - origin_y
                r = int(y // self.grid_size)
                c = int(x // self.grid_size)
                valid = self.check_piece(r, c)
                if valid:
                    self.action = (self.piece, r, c)
                    return self.action

    def check_piece(self, r, c):
        return self.board[r][c] == '.'

    def set_piece(self, r, c):
        if self.board[r][c] == '.':
            self.board[r][c] = self.piece

            if self.piece == GameBoard.PIECE_B:
                self.piece = GameBoard.PIECE_W
            else:
                self.piece = GameBoard.PIECE_B

            return True
        return False

    def check_win(self, r, c):
        n_count = self.get_continuous_count(r, c, -1, 0)
        s_count = self.get_continuous_count(r, c, 1, 0)

        e_count = self.get_continuous_count(r, c, 0, 1)
        w_count = self.get_continuous_count(r, c, 0, -1)

        se_count = self.get_continuous_count(r, c, 1, 1)
        nw_count = self.get_continuous_count(r, c, -1, -1)

        ne_count = self.get_continuous_count(r, c, -1, 1)
        sw_count = self.get_continuous_count(r, c, 1, -1)

        if (n_count + s_count + 1 >= self.connect_num) or (e_count + w_count + 1 >= self.connect_num) or \
                (se_count + nw_count + 1 >= self.connect_num) or (ne_count + sw_count + 1 >= self.connect_num):
            self.winner = self.board[r][c]
            self.game_over = True

    def get_continuous_count(self, r, c, dr, dc):
        piece = self.board[r][c]
        result = 0
        i = 1
        while True:
            new_r = r + dr * i
            new_c = c + dc * i
            if 0 <= new_r < self.board_num and 0 <= new_c < self.board_num:
                if self.board[new_r][new_c] == piece:
                    result += 1
                else:
                    break
            else:
                break
            i += 1
        return result

    def draw(self, screen):
        pygame.draw.rect(screen, (185, 122, 87),
                         [self.start_x - self.edge_size, self.start_y - self.edge_size,
                          (self.board_num - 1) * self.grid_size + self.edge_size * 2, (self.board_num - 1) * self.grid_size + self.edge_size * 2], 0)

        for r in range(self.board_num):
            y = self.start_y + r * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [self.start_x, y], [self.start_x + self.grid_size * (self.board_num - 1), y], 2)

        for c in range(self.board_num):
            x = self.start_x + c * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [x, self.start_y], [x, self.start_y + self.grid_size * (self.board_num - 1)], 2)

        for r in range(self.board_num):
            for c in range(self.board_num):
                piece = self.board[r][c]
                if piece != '.':
                    if piece == 'b':
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)

                    x = self.start_x + c * self.grid_size
                    y = self.start_y + r * self.grid_size
                    pygame.draw.circle(screen, color, [x, y], self.grid_size // 2)


if __name__ == '__main__':
    game = GameBoard()
    while game.going:
        game.next_step()

    pygame.quit()

