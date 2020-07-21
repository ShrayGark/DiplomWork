import functions as f


class Symbol:
    def __init__(self, data):
        self.center_x = data['center_x']
        self.center_y = data['center_y']
        self.height = data['height']
        self.weight = data['weight']
        self.symbol = data['symbol']
        self.cls = 0
        self.dependencies = {}

    def __str__(self):
        return f'center: {self.center_x, self.center_y}, height: {self.height}, weight: {self.weight},' \
               f' symbol: {self.symbol}, class: {self.cls}'

    def __lt__(self, other):
        return self.center_x < other.center_x

    def __repr__(self):
        return self.symbol


class Level:
    def __init__(self, height, area):
        self.symbols = []
        self.height = height
        self.area = area

    def add_symbol(self, symbol):
        if symbol not in self.symbols:
            # print(self.height, self.area)
            # print(f'{self.height - self.area} <= {symbol.center_y} <= {self.height + self.area}')
            if self.height - self.area / 4 <= symbol.center_y <= self.height + self.area / 4:
                # print('IN RECT')
                self.symbols.append(symbol)
                return True
            else:
                return False
        else:
            return False

    def __repr__(self):
        buffer = ''
        for s in self.symbols:
            buffer += f' {s.symbol}'
        return buffer

    def split_to_lines(self, area):
        res = []
        buffer = [self.symbols[0]]
        pref_x = self.symbols[0].center_x
        for i in range(1, len(self.symbols)):
            if self.symbols[i].center_x <= pref_x + area * 3:
                buffer.append(self.symbols[i])
                pref_x = self.symbols[i].center_x
            else:
                res.append(buffer)
                buffer = list()
                buffer.append(self.symbols[i])
                pref_x = self.symbols[i].center_x
        if buffer:
            res.append(buffer)
        return res


class ExpLine:
    def __init__(self, symbols):
        self.symbols = symbols
        self.str_symbols = ''
        self.center_x = 0
        self.center_y = 0
        self.height = 0
        self.weight = 0
        self.dependence = False
        for symbol in self.symbols:
            self.str_symbols += symbol.symbol
            self.center_x += symbol.center_x
            self.center_y += symbol.center_y
            self.height += symbol.height
            self.weight += symbol.weight
        self.center_x /= len(self.symbols)
        self.center_y /= len(self.symbols)
        self.height /= len(self.symbols)
        self.left_x = self.symbols[0].center_x
        self.left_y = self.symbols[0].center_y

    def __repr__(self):
        buffer = ''
        for s in self.symbols:
            buffer += f' {s.symbol}'
        buffer += f' {self.dependence}'
        return buffer

    def add_symbol(self, symbol):
        self.symbols.append(symbol)

    def get_exp(self):
        res = ''
        for symbol in self.symbols:
            if symbol.cls == 1:
                if 'up' in list(symbol.dependencies.keys()):
                    up = symbol.dependencies['up'].get_exp()
                    down = symbol.dependencies['down'].get_exp()
                    res += f'{up}/{down}'
                    continue
            elif 'pow' in list(symbol.dependencies.keys()):
                power = symbol.dependencies['pow'].get_exp()
                res += f'{symbol.symbol}**{power}'
                continue
            res += symbol.symbol
        res = '(' + res + ')'
        return res


class SymbolKeeper:
    def __init__(self, data):
        self.symbols = []
        self.operations = []
        self.numbers = []
        self.levels = []
        self.exp_lines = []
        for d in data:
            self.symbols.append(Symbol(d))

        self.h_range = 1
        self.v_range = 1

    def print_levels(self):
        for level in self.levels:
            print(level)

    def print_exp_lines(self):
        for line in self.exp_lines:
            print(line)

    def print_data(self):
        for symbol in self.symbols:
            print(symbol)

    def sort_symbols(self):
        self.symbols.sort()

    def classified_symbols(self):
        number = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}  # 0
        operation = {'+', '-'}  # 1
        variable = {'x', 'y'}  # 2
        for symbol in self.symbols:
            if symbol.symbol in number:
                self.numbers.append(symbol)
                symbol.cls = 0

            if symbol.symbol in operation:
                self.operations.append(symbol)
                symbol.cls = 1

            if symbol.symbol in variable:
                self.numbers.append(symbol)
                symbol.cls = 2

    def count_hv_range(self):
        buffer_x = 0
        buffer_y = 0
        for number in self.numbers:
            buffer_x += number.weight
            buffer_y += number.height
        self.v_range = buffer_x / len(self.numbers)
        self.h_range = buffer_y / len(self.numbers)

    def resolve_uncertainty(self):
        self.count_hv_range()
        for operation in self.operations:
            if operation.symbol == '-':
                # print('IN -')
                counter = 0
                for number in self.numbers:
                    if operation.center_x - operation.weight / 2 <= number.center_x <= operation.center_x + operation.weight / 2 \
                            and operation.center_y >= number.center_y >= operation.center_y - number.height * 2:
                        # print(number.symbol)
                        counter += 1

                    if operation.center_x - operation.weight / 2 <= number.center_x <= operation.center_x + operation.weight / 2 \
                            and operation.center_y <= number.center_y <= operation.center_y + number.height * 2:
                        # print(number.symbol)
                        counter += 1
                if counter >= 2:
                    # print('WORK')
                    operation.symbol = '_'

    def level_split(self):
        for symbol in self.symbols:
            if self.levels:
                flag = True
                for level in self.levels:
                    if level.add_symbol(symbol):
                        flag = False
                        break
                if flag:
                    self.levels.append(Level(symbol.center_y, self.v_range))
                    self.levels[-1].add_symbol(symbol)

            else:
                self.levels.append(Level(symbol.center_y, self.v_range))
                self.levels[0].add_symbol(symbol)

    def build_exp_line(self):
        res = []
        for level in self.levels:
            for line in level.split_to_lines(self.h_range):
                res.append(line)

        for line in res:
            self.exp_lines.append(ExpLine(line))

    def find_dependence(self):
        for symbol in self.symbols:
            for line in self.exp_lines:
                if symbol.symbol == '_':
                    if symbol.symbol not in line.str_symbols:
                        if symbol.center_x - symbol.weight/2 < line.center_x < symbol.center_x + symbol.weight/2:
                            if symbol.center_y > line.center_y > symbol.center_y - line.height * 2:
                                symbol.dependencies['up'] = line
                                line.dependence = True

                            if symbol.center_y < line.center_y < symbol.center_y + line.height * 2:
                                symbol.dependencies['down'] = line
                                line.dependence = True
                elif symbol.cls == 0:
                    right_up_angle_x = symbol.center_x + symbol.weight/2
                    right_up_angle_y = symbol.center_y - symbol.height/2
                    if right_up_angle_x < line.left_x < right_up_angle_x + symbol.weight:
                        if right_up_angle_y > line.left_y > right_up_angle_y - symbol.height:
                            symbol.dependencies['pow'] = line
                            line.dependence = True

    def build_exp(self):
        res = 0
        for line in self.exp_lines:
            if not line.dependence:
                res = line.get_exp()
                break

        print(f'Выражение: {res}')
        print(f'Ответ: {eval(res)}')

