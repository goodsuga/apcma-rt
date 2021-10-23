# Код написан командой RubEx_Tech для хакатона "Цифровой Прорыв" в направлении "Транспорт и логистика"
# Кейс: Шереметьево

import pandas
import time
from datetime import datetime, timedelta
from itertools import combinations
import random
from tqdm import tqdm
import numpy
from warnings import simplefilter
simplefilter(action="ignore", category=pandas.errors.PerformanceWarning)


def parse_csv(file, sep, decimal):
    """Для более удобного считывания .csv-файла в пандас"""
    return pandas.read_csv(file, sep=sep, decimal=decimal)


def get_plane_class_by_capacity(capacity):
    """
    Функция классификации самолета по макс. числу мест
    Необходима для правильного определения времени обслуживания
    """
    if capacity <= 120:
        return "Regional"
    if capacity > 120 and capacity <= 220:
        return "Narrow_Body"
    else:
        return "Wide_Body"


def parse_date(datestr):
    """
    Превращаем строку вида ГГГГ-ММ-ДД ЧЧ:ММ:CC в питоновский datetime
    """
    components = datestr.split(" ")
    _date = components[0].split(".")
    _time = components[1].split(":")
    return datetime(int(_date[2]), int(_date[1]), int(_date[0]), int(_time[0]), int(_time[1]))


# run_trial - это и есть алгоритм. Исполнение кода начинается с вызова этой функции
def run_trial(aircraft_stands_file, handling_rates_file, handling_time_file, timetable):
    
    # Парсим все необходимые файлы: характеристики мест стоянок, тарифы, время обслуживания и расписание
    stands = parse_csv(aircraft_stands_file, ",", ".")
    costs = parse_csv(handling_rates_file, ",", ".")
    handle_times = parse_csv(handling_time_file, ",", ".")
    planes = parse_csv(timetable, ",", ".")

    # Для гибкости и удобства дальнейших расчетов данные необходимо модифицировать,
    # добавив некоторые колонки
    # 1) Умножим время рулежки на место стоянки на тариф стоимости рулежки
    # чтобы получить чистую стоимость рулежки ВС до этого места стоянки (МС)
    taxiing_cost = costs[costs['Name']=='Aircraft_Taxiing_Cost_per_Minute']['Value'].values[0]
    stands['Taxiing_Cost'] = stands['Taxiing_Time'].apply(lambda taxi_time: taxi_time*taxiing_cost)
    
    # 2) Проделаем то же самое для автобусов - умножим время их пути до каждого терминала
    # на тариф стоимости автобуса за минуту
    bus_cost = costs[costs['Name']=='Bus_Cost_per_Minute']['Value'].values[0]
    for terminal in range(1, 6):
        stands[f"{terminal}"] = stands[f"{terminal}"].apply(lambda bus_time: bus_time*bus_cost)
        stands.rename({f"{terminal}": f"Bus_Cost_To_{terminal}"}, inplace=True, axis='columns')

    
    # Не останавливаясь на одних лишь точках, дополним данные и о самолетах
    # 1) Определим для каждого самолета его класс. Класс определяется по макс. числу мест
    # предусмотренных в конструкции.
    planes['CLASS'] = planes['flight_AC_PAX_capacity_total'].apply(get_plane_class_by_capacity)
    
    # 2) Для дальнейших вычислений немного изменим индексирование таблицы с временем обслуживания судна,
    # чтобы разблокировать возможность делать так: handle_times.loc[класс_самолета]
    # т.е. выбирать время обслуживания по классу
    handle_times.set_index("Aircraft_Class", drop=True, inplace=True)

    # 3) Загрузим тарифы на минуту обслуживания на автобусах и телетрапе
    away_stand_cost_per_minute = costs[costs['Name'] == 'Away_Aircraft_Stand_Cost_per_Minute']['Value']
    bridge_stand_cost_per_minute = costs[costs['Name'] == 'JetBridge_Aircraft_Stand_Cost_per_Minute']['Value']

    # 4) Теперь произведем расчеты стоимости обслуживания каждого самолета на телетрапе, автобусах, 
    # и также (что очень немаловажно!) вычислим тариф использования места с телетрапом НО БЕЗ телетрапа
    # т.е. когда телетрап на МС есть, но ВС обслуживается автобусами
    planes['HANDLING_COST_BRIDGE'] = planes['CLASS'].apply(lambda plane_class: handle_times.loc[plane_class]['JetBridge_Handling_Time']*bridge_stand_cost_per_minute)
    planes['HANDLING_COST_AWAY'] = planes['CLASS'].apply(lambda plane_class: handle_times.loc[plane_class]['Away_Handling_Time']*away_stand_cost_per_minute)
    planes['HANDLING_COST_BRIDGE_ON_BUSES'] = planes['CLASS'].apply(lambda plane_class: handle_times.loc[plane_class]['Away_Handling_Time']*bridge_stand_cost_per_minute)

    # 5) Начинается одна из самых важных частей алгоритма. В зависимости от того, как обслуживается самолет, он занимает
    # и освобождает точку в разное время. Так, если самолет отправляется (flight_AD == 'D'), то чтобы понять, когда самолет
    # занимает МС необходимо из времени его вылета вычесть время на обслуживание и рулежку
    # Если самолет прибывает, то чтобы определить время когда самолет освободит точку нужно наоборот прибавить
    # время обслуживания и рулежки ко времени его прилета
    # Но и это еще не все. Мало того, что нужно учитывать отправляется ли самолет или прибывает,
    # нужно помнить, что время на обслуживание самолета на автобусах и телетрапе разнится. 
    # Поэтому каждому самолету нужно 4 дополнительных свойства для удобства дальнейших операций:
    # a) Stand_Take_Time_Away - время занятия самолетом выбранного места стоянки, при условии что обслуживаться самолет будет автобусами
    # b) Stand_Take_Time_Bridge - время занятия самолетом выбранного места стоянки, при условиии что обслуживаться самолет будет телетрапом
    # c) Stand_Leave_Time_Bridge - время освобождения самолетом выбранного места стоянки, при условии, что обслуживаться самолет будет телетрапами
    # d) Stand_Leave_Time_Away - время освобождения самолетом выбранного места стоянки, при условии, что обслуживаться самолет будет телетрапами
    planes['Stand_Take_Time_Away'] = planes['flight_datetime'].apply(parse_date)
    planes.sort_values(by='Stand_Take_Time_Away', inplace=True) # Для удобства отсортируем датасет по flight_datetime. Самые ранние рейсы - вперед
    planes.reset_index(drop=True, inplace=True)

    # Пусть по умолчанию все 4 вида времени будут равны flight_datetime
    planes['Stand_Take_Time_Bridge'] = planes['Stand_Take_Time_Away'].copy(deep=True)
    planes['Stand_Leave_Time_Bridge'] = planes['Stand_Take_Time_Away'].copy(deep=True)
    planes['Stand_Leave_Time_Bridge'] = planes['Stand_Take_Time_Away'].copy(deep=True)

    # Самолет отправляется, следовательно время занятия им МС равно flight_datetime - handling
    # ВАЖНО! В ходе работы по кейсу выяснилось, что в расчет следует включать и время рулежки
    # Недостаток этого элемента в расчете будет исправлен дальше в коде
    planes.loc[planes['flight_AD']=='D', 'Stand_Take_Time_Away'] = planes.loc[planes['flight_AD'] == 'D', 'Stand_Take_Time_Away']-\
        planes.loc[planes['flight_AD'] == 'D', 'CLASS'].apply(lambda plane_class: timedelta(minutes=int(handle_times.loc[plane_class]['Away_Handling_Time'])))

    # Самолет отправляется, время занятия МС с телетрапом равно flight_datetime - handling
    planes.loc[planes['flight_AD']=='D', 'Stand_Take_Time_Bridge'] = planes.loc[planes['flight_AD'] == 'D', 'Stand_Take_Time_Bridge']-\
        planes.loc[planes['flight_AD'] == 'D', 'CLASS'].apply(lambda plane_class: timedelta(minutes=int(handle_times.loc[plane_class]['JetBridge_Handling_Time'])))

    # Самолет отправляется, следовательно время освобождения им МС равно времени его отправления, т.е.
    # равно flight_datetime
    planes.loc[planes['flight_AD']=='D', 'Stand_Leave_Time_Away'] = planes.loc[planes['flight_AD'] == 'D', 'flight_datetime'].apply(parse_date)

    # Самолет прибывает, следовательно время освобождения им МС на автобусах = время прибытия + handling на автобусах
    planes.loc[planes['flight_AD']=='A', 'Stand_Leave_Time_Away'] = planes.loc[planes['flight_AD']=='A', 'Stand_Take_Time_Away']+\
        planes.loc[planes['flight_AD'] == 'A', 'CLASS'].apply(lambda plane_class: timedelta(minutes=int(handle_times.loc[plane_class]['Away_Handling_Time'])))

    # Самолет прибывает, следовательно время освобожления им МС на телетрапе = время прибытия + handling на телетрапе 
    planes.loc[planes['flight_AD']=='A', 'Stand_Leave_Time_Bridge'] = planes.loc[planes['flight_AD']=='A', 'Stand_Take_Time_Bridge']+\
        planes.loc[planes['flight_AD'] == 'A', 'CLASS'].apply(lambda plane_class: timedelta(minutes=int(handle_times.loc[plane_class]['JetBridge_Handling_Time'])))


    # Переиндексируем таблицу мест стоянок так, чтобы открыть доступ к
    # stands.iloc[непосредственный, реальный номер места стоянки]
    # Так лучше и удобнее, чем оперировать индексом номера места стоянки в таблице мест стоянок
    stands.set_index("Aircraft_Stand", drop=True, inplace=True)

    # 6) Готовимся к еще одному ключевому моменту алгоритма.
    # Для адекватной оценки стоимости занятия самолетом той или иной точки необходимо
    # просчитать стоимость постановки на точку на автобусах и на телетрапе.
    # Чтобы принять в расчет телетрап, нам необходимо соблюсти два условия:
    # a) Терминал точки должен быть равен терминалу, в котором взлетает/садится самолет
    # b) Тип рейса (международный или внутренний, I/D - International, Domestic) должен совпадать с
    # типом, указанным в описании точки
    # Чтобы легко обрабатывать разные комбинации точек и бортов, введем понятия "Кода телетрапа" -
    # BRIDGE_CODE. BRIDGE_CODE точки - это строка вида "TATD", где T - это код доступности терминала
    # (N = No, I = International, D = Domestic). 
    # Bridge code рейса - это строка вида "TF", где T - это flight_ID (I/D), а F - это flight_AD (A/D)
    # Соответственно, если прибывает международный рейс, то его BRIDGE_CODE = IA
    # Внутренний рейс отправляется = DD
    # Сопоставляя BRIDGE_CODE точки и рейса можно узнать, совместим ли данный рейс с терминалом
    # И, пользуясь полученной информацией, вычислить тариф
    stands['JetBridge_on_Arrival'].fillna("N", inplace=True)
    stands['JetBridge_on_Departure'].fillna("N", inplace=True)
    stands['JetBridge_on_Arrival'] = stands['JetBridge_on_Arrival'].astype(str)
    stands['JetBridge_on_Departure'] = stands['JetBridge_on_Departure'].astype(str)
    stands['BRIDGE_CODE'] = stands['JetBridge_on_Arrival']+stands['JetBridge_on_Departure']
    stands['BRIDGE_CODE'] = stands['BRIDGE_CODE'].apply(lambda code: code[0]+"A"+code[1]+"D")

    # Теперь bridge_code для самолетов
    planes['BRIDGE_CODE'] = planes['flight_ID']+planes['flight_AD']


    # Всё готово к главному подготовительному шагу. Расчет стоимости постановки каждого самолета на каждую точку
    # Мы трактуем стоимость однозначно: Стоимость занятия рейсом места стоянки - это минимальная стоимость из всех вариантов его обслуживания
    # Иными словами, рейс может встать на точку и стоимость его обслуживания будет определяться на автобусах обслуживание идет
    # или на телетрапах. Однако для дальнейшей простоты и эффективности мы всегда считаем, что стоимость - это минимальный вариант
    # На телетрапе дешевле? Значит стоимость постановки рейса на эту точку = стоимости обслуживания телетрапом
    # На автобусах дешевле? Значит стоимость постановки рейса на эту току = стоимости обслуживания автобусом
    # Ремарка: в ходе работы по кейсу выяснилось, что если рейс может использовать телетрап, но при этом выбирает обслуживаться автобусами
    # То время обслуживания на автобусах нужно умножать не на автобусный тариф, а на тариф телетрапа. 
    # Ремарка 2: ранее уже было оговорено, что стоимость рулежки нужно включать в расчет времени занятия/освобожления 
    # воздушным судном. Блок кода ниже - лучшее место, чтобы это сделать, так как время рулежки определяется не рейсом, а точкой.
    # Наконец, пояснения по создаваемым матрицам:
    # a) cost_matrix - матрица стоимостей. "Сколько стоит поставить каждый самолет на каждую точку"
    # b) take_matrix - матрица времени занятия самолетом точки. "Во сколько каждый самолет займет каждую точку"
    # c) leave_matrix - матрица времени освобождения самолетом точки. "Во сколько каждый самолет освободит каждую точку"
    # d) comment_matrix - матрица типа обслуживания, при котором достигается стоимость, время занятия и освобождения
    cost_matrix = pandas.DataFrame()
    take_matrix = pandas.DataFrame()
    leave_matrix = pandas.DataFrame()
    comment_matrix = pandas.DataFrame()
    for index, point in stands.iterrows(): # INDEX = это не порядковый, а именно настоящий номер места стоянки
        # Если телетрап доступен на точке, то обслуживать на автобусах будем по тарифу телетрапа
        if point['JetBridge_on_Arrival'] != 'N' or point['JetBridge_on_Departure'] != 'N': 
            wo_bridge = point['Taxiing_Cost'] +\
                                                            planes['HANDLING_COST_BRIDGE_ON_BUSES'] +\
                                                            planes['flight_terminal_#'].apply(lambda terminal: point[f'Bus_Cost_To_{terminal}'])*((1.0+planes['flight_PAX']/80.0).apply(int))
        else: # Если телетрапа нет, то на автобусах обслуживаем по тарифу автобусов
            wo_bridge = point['Taxiing_Cost'] +\
                                                            planes['HANDLING_COST_AWAY'] +\
                                                            planes['flight_terminal_#'].apply(lambda terminal: point[f'Bus_Cost_To_{terminal}'])*((1.0+planes['flight_PAX']/80.0).apply(int))
        
        # Расчитаем стоимость обслуживания на терминале. Если терминал недоступен по тем или иным причинам,
        # то сначала по приведенной ниже формуле стоимость обслуживания на нем будет равна нулю. Чтобы программа никогда не подумала, что это
        # и есть самый оптимальный вариант придется прибегнуть к методам костыльно-ориентированного программирования
        # Если стоимость обслуживания на телетрапе равна нулю, то изменить её на 2^32. Всегда найдется вариант лучше, чем
        # стоимость в 2^32!
        with_bridge = (point['Taxiing_Cost'] + planes['HANDLING_COST_BRIDGE'])*(planes['BRIDGE_CODE'].apply(
            lambda code: code in point['BRIDGE_CODE']) & (planes['flight_terminal_#'] == point['Terminal'])).apply(int)
        with_bridge = with_bridge.apply(lambda cost: cost if cost > 0 else 2**32)

        # Как и было оговорено, стоимость занятия точки равна минимальной стоимости из двух возможных вариантов:
        # С телетрапом или с автобусами
        cost_matrix[index] = list(map(min, zip(wo_bridge.values, with_bridge.values)))

        # Определяем время занятия и освобождения точки, в зависимости от минимального тарифа.
        # Сюда же комментарий
        take_matrix[index] = [bridge_take_time if bridge_cost <= bus_cost else bus_take_time for bridge_take_time,bus_take_time,bridge_cost,bus_cost\
            in zip(planes['Stand_Take_Time_Bridge'], planes['Stand_Take_Time_Away'], with_bridge.values, wo_bridge.values)]
        leave_matrix[index] = [bridge_leave_time if bridge_cost <= bus_cost else bus_leave_time for bridge_leave_time,bus_leave_time,bridge_cost,bus_cost\
            in zip(planes['Stand_Leave_Time_Bridge'], planes['Stand_Leave_Time_Away'], with_bridge.values, wo_bridge.values)]
        comment_matrix[index] = ["Телетрап" if bridge_cost <= bus_cost else "Автобусы" for bridge_cost,bus_cost\
            in zip(with_bridge.values, wo_bridge.values)]

    # Сейчас наши матрицы вида "ТОЧКА: САМОЛЕТ1, САМОЛЕТ2, САМОЛЕТ3...."
    # Для удобства дальнейшей работы их нужно транспонировать в вид
    # "Самолет1: точка1, точка2, точка3, точка4, точка5...."
    cost_matrix = cost_matrix.T
    take_matrix = take_matrix.T
    leave_matrix = leave_matrix.T
    comment_matrix = comment_matrix.T

    # Вишенка на торте! Главная фича! Основа собственно оптимизационного алгоритма
    # Для каждого самолета всю информацию (стоимость, время занятия, освобождения, комментарий)
    # отсортируем по стоимости занимаемых точек.
    # То есть cost_lines[самолет номер 1][0] будет содержать самую минимальную стоимость постановки самолета на точку
    # cost_lines[самолет номер 1][2] будет содержать уже менее выгодный вариант, и так далее, по возрастанию стоимости
    # Итак, для каждого самолета у нас есть "линии" - стоимости, номеров точек, времени занятия и освобождения и т.д.
    # и все они отсортированы наиболее выгодным для самолета образом. Удобство такого подхода будет продемонстрировано ниже
    cost_lines = {}
    point_lines = {}
    take_lines = {}
    leave_lines = {}
    comment_lines = {}
    for plane in range(len(planes)):
        sorted_vals = cost_matrix[plane].sort_values()
        cost_lines[plane] = sorted_vals.tolist()
        point_lines[plane] = list(sorted_vals.index)
        # Добавим к расчету времени занятия/освобождения Taxiing_Time
        if planes.iloc[plane]['flight_AD'] == 'D':
            take_lines[plane] = (take_matrix.loc[sorted_vals.index, plane]-stands.loc[sorted_vals.index, 'Taxiing_Time'].apply(lambda tt: timedelta(minutes=tt))).tolist()
            leave_lines[plane] = leave_matrix.loc[sorted_vals.index, plane].tolist()
        else:
            take_lines[plane] = take_matrix.loc[sorted_vals.index, plane].tolist()
            leave_lines[plane] = (leave_matrix.loc[sorted_vals.index, plane]+stands.loc[sorted_vals.index, 'Taxiing_Time'].apply(lambda tt: timedelta(minutes=tt))).tolist()
        
        comment_lines[plane] = comment_matrix.loc[sorted_vals.index, plane].tolist()
    
    # Пользуемся фишкой расстановки по возрастанию стоимости. Нулевая по индексу точка для любого самолета - 
    # это точка с минимальной стоимостью размещения. Чем дальше точка в списке - тем она дороже
    planes['Aircraft_Stand'] = list(map(lambda i: point_lines[i][0], range(len(planes))))
    from copy import deepcopy
    minimals = deepcopy([cost_lines[plane_index][0] for plane_index in range(len(planes))])
    #del cost_matrix, take_matrix, leave_matrix, comment_matrix

    # Багфикс из будущего: в ходе работы над кейсом выяснилось, что
    # WIDE_BODY суда конфликтуют не только, когда стоят на соседних местах с телетрапами в одно время,
    # А еще и когда терминалы выбранных точек совпадают. Добавим эту строчку, чтобы позже внедрить поправку
    stand_to_terminal = dict([(index, terminal) for index, terminal in zip(list(stands.index), stands['Terminal'])])
    
    # Две оперативные колонки - непосредственное время освобождения точки и занятия точки.
    # Актуальны только для текущей, выбранной самолетом точки
    planes['ACTUAL_LEAVE'] = [leave_lines[i][0] for i in range(len(planes))]
    planes['ACTUAL_TAKE'] = [take_lines[i][0] for i in range(len(planes))]
    plane_intersections = {}
    for plane_index in range(len(planes)):
        plane = planes.iloc[plane_index]
        plane_intersections[plane_index] = list(planes.loc[(planes.index != plane_index) & (((planes['Stand_Take_Time_Away'] < plane['Stand_Leave_Time_Away'])\
            & (planes['Stand_Take_Time_Away'] >= plane['Stand_Take_Time_Away'])) | ((planes['Stand_Take_Time_Bridge'] < plane['Stand_Leave_Time_Bridge'])\
            & (planes['Stand_Take_Time_Bridge'] >= plane['Stand_Take_Time_Bridge'])))].index)
    
    # Вспомогательные элементы для работы с pandas
    collist = list(planes.columns)
    aircraft_stand_index = collist.index("Aircraft_Stand")
    actual_leave_index = collist.index("ACTUAL_LEAVE")
    actual_take_index = collist.index("ACTUAL_TAKE")

    # Наконец, сам алгоритм.
    # К началу цикла ниже все самолеты уже вслепую, без проверки конфликтов, расставлены
    # на самые идеальные для них точки.
    # задача цикла - найти конфликты самолетов и исправить их таким образом,
    # чтобы менее всего увеличить издержки для аэропорта.
    # Когда цикл завершится в расстановке самолетов гарантировано не будет конфликтов
    # Вернее сказать даже, что цикл завершится только при условии отсутствия конфликтов в расстановке
    # судов.
    while True:
        conflicts_found = False

        for i in range(len(planes.index)):  # Конфликты определять для каждого самолета
            plane = planes.iloc[i]
            # Самолеты могут считаться конфликтующими с текущим ЕСЛИ:
            # 1. ИНДЕКС САМОЛЕТА != ИНДЕКСУ ТЕКУЩЕГО САМОЛЕТА __ И __ 
            # 2. ЕСЛИ САМОЛЕТ ЗАНИМАЕТ МЕСТО СТОЯНКИ ПОЗЖЕ ТЕКУЩЕГО САМОЛЕТА, НО ДО МОМЕНТА ОСВОБОЖДЕНИЯ
            # МЕСТА СТОЯНКИ ТЕКУЩИМ САМОЛЕТОМ __ И __
            # 3. МЕСТО СТОЯНКИ САМОЛЕТА СОВПАДАЕТ С МЕСТОМ СТОЯНКИ ТЕКУЩЕГО __ ИЛИ __
            # 4. МЕСТО СТОЯНКИ САМОЛЕТА ОТЛИЧАЕТСЯ ОТ МЕСТА СТОЯНКИ ТЕКУЩЕГО ПО ИНДЕКСУ НА ЕДИНИЦУ __ ПРИ УСЛОВИИ __
            # Что класс текущего самолета и второго самолета = 'Wide_Body' (широкофюзеляжный) __ И __ терминалы точек самолетов совпадают
                    
            conflicts =  planes.loc[((planes.index != i)\
            & (planes['ACTUAL_TAKE'] <= plane['ACTUAL_LEAVE'])\
            & (planes['ACTUAL_TAKE'] >= plane['ACTUAL_TAKE']))\
            & ((planes['Aircraft_Stand'] == plane['Aircraft_Stand'])\
            | ((abs(planes['Aircraft_Stand']-plane['Aircraft_Stand']) == 1)\
            & (planes['CLASS']=='Wide_Body') & (plane['CLASS']=='Wide_Body')\
            & (planes['Aircraft_Stand'].apply(stand_to_terminal.get) == stand_to_terminal[plane['Aircraft_Stand']])))]
            
            # Если конфликты были найдены, исправляем их
            if len(conflicts) > 0: 
                conflicts_found = True

                # делаем список индексов конфликтующих самолетов
                conflicting_plane_indecies = [i]+list(conflicts.index)

                # Делаем все сочетания решений конфликта.
                # Решение конфликта, в данном случае, - это набор самолетов, которых мы сдвинем на следующие для них оптимальные точки
                # Сразу же определим стоимости решений. Стоимость решения - это сумма разниц стоимости размещения самолета
                # на следующей для него оптимальной точке и размещении его на текущей точке
                # Лучшее решение - это такое решение, реализация которого обеспечит аэропорту наименьший рост издержек
                # Ремарка: в теории, все должно было работать именно так. На практике же оказалось, что 
                # может существовать несколько оптимальных решений с одинаковой стоимостью.
                # Именно в этих трех строках и лежит недетерминированная природа алгоритма: из 
                # нескольких оптимальных решений с одинаковой стоимостью алгоритм выбирает случайное
                # Эта же особенность позволяет применять экстенсивное расширение ресурсов для улучшения работы алгоритма:
                # Чем больше процессов одновременно считают различные комбинации, тем выше шанс того, что удастся выйти
                # на все более и более оптимальное решение. 
                solutions = numpy.array(list(combinations(conflicting_plane_indecies, len(conflicting_plane_indecies)-1)))
                solution_costs = numpy.array([sum([cost_lines[plane][1]-cost_lines[plane][0] for plane in solution]) for solution in solutions])
                best_solution = random.choice(solutions[solution_costs==solution_costs.min()]) #random.choice(solutions[:7])
                # Решение было определено.
                # Чтобы его "внедрить" надо всем передвигаемым самолетам обновить ACTUAL_LEAVE, ACTUAL_TAKE, Aircraft_Stand
                
                planes.iloc[best_solution, [aircraft_stand_index, actual_leave_index, actual_take_index]] = [[point_lines[plane_index][1], 
                                                                                                            leave_lines[plane_index][1], 
                                                                                                            take_lines[plane_index][1]] for plane_index in best_solution]
                # Для каждого самолета, чтобы не мучаться с хранением индексов, введем и будем поддерживать правило:
                # в любой линии для любого самолета элемент под индексом ноль будет для самолета актуальным
                # иными словами, cost_lines[plane][0] - это всегда текущая стоимость размещения самолета,
                # cost_lines[plane][1] - это всегда стоимость размещения самолета на следующей для него оптимальной точке
                # Та же логика применяется и для других линий - cost_lines, point_lines, leave_lines, take_lines, comment_lines
                for plane in best_solution:
                    del point_lines[plane][0], leave_lines[plane][0], take_lines[plane][0], cost_lines[plane][0]
                    del comment_lines[plane][0]

        # Если среди всех самолетов не был обнаружен ни один конфликт, то выходим из цикла
        if not conflicts_found:
            break

    # Добавление вспомогательных колонок для быстрого анализа выходного решения
    planes['Стоимость размещения'] = [cost_lines[plane][0] for plane in range(len(planes))]
    planes['Минимальная стоимость размещения'] = minimals
    planes['Тип обслуживания'] = [comment_lines[plane][0] for plane in range(len(planes))] 
    return planes
    

# Функция анализа "Что-если": на вход
# принимает два набора файлов с различающимися характеристиками и проводит их сравнение
# по метрике совокупной стоимости размещения самолетов.
# Используя эту функцию можно узнать, например, какую экономию средств обеспечит строительство дополнительного
# места стоянки, к какому изменению стоимостей размещения приведет изменение тарифов или сокращение времени обслуживания
# воздушного судна.
def what_if(aircraft_stands_file, handling_rates_file, handling_time_file, timetable,
            aircraft_stands_file_2, handling_rates_file_2, handling_time_file_2, timetable_2, n_proc):
    results_1 = []
    results_2 = []

    pool = Pool(n_proc)
    for i in range(n_proc):
        pool.apply_async(run_trial, args=(aircraft_stands_file, handling_rates_file, handling_time_file, timetable), callback=results_1.append)
    pool.close()
    pool.join()
    results_1 = list(sorted(results_1, key=lambda res: res['Стоимость размещения'].sum()))[0]

    pool = Pool(n_proc)
    for i in range(n_proc):
        pool.apply_async(run_trial, args=(aircraft_stands_file_2, handling_rates_file_2, handling_time_file_2, timetable_2), callback=results_2.append)
    pool.close()
    pool.join()
    results_2 = list(sorted(results_2, key=lambda res: res['Стоимость размещения'].sum()))[0]
    r1 = results_1['Стоимость размещения'].sum()
    r2 = results_2['Стоимость размещения'].sum()
    print(f"Стоимость размещения в варианте 1: {r1}; В варианте 2 = {r2}; R1/R2 = {round(r1*100/r2, 4)}%")
    return results_1, results_2


# Функция отрисовки диаграммы ганта. 
# Отрисует все самолеты, фигурирующие в planes
# Для работы в датафрейме нужны поля ACTUAL_TAKE - время занятия самолетом места стоянки
# ACTUAL_LEAVE - время освобождения самолетом места стоянки
# И Aircraft_Stand - собственно номера точек
import plotly.express as px
def plot_gantt(planes):
    fig = px.timeline(planes, x_start="ACTUAL_TAKE", x_end="ACTUAL_LEAVE", y="Aircraft_Stand", color="Aircraft_Stand")
    fig.show()


# Запуск кода происходит тут.
# Ищем оптимальное решение сразу в нескольких процессах
from pprint import pprint
from multiprocessing import Pool

if __name__ == '__main__':
    run_trial("Aircraft_Stands_Private.csv", "Handling_Rates_Private.csv", "Handling_Time_Private.csv", "Timetable_private.csv")
    for i in range(1000):
        sgs = time.time()
        results = []
        n_proc = 3
        pool = Pool(n_proc)
        for i in range(n_proc):
            pool.apply_async(run_trial, args=("Aircraft_Stands_Private.csv", "Handling_Rates_Private.csv", "Handling_Time_Private.csv", "Timetable_private.csv"), 
            callback=results.append)
        pool.close()
        pool.join()
        res_sum = [(index, results[index]['Стоимость размещения'].sum()) for index in range(len(results))]
        res_sum = list(sorted(res_sum, key=lambda res: res[1]))
        _r = list(sorted(results, key=lambda res: res['Стоимость размещения'].sum()))[0]
        _r.to_excel(f"{_r['Стоимость размещения'].sum()}.xlsx", index=False)
        pprint(res_sum)
        print("Работа завершена")
        print(f"Ушло {round(time.time()-sgs, 2)}")