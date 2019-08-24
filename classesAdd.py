import sys

class MyClass():
    def __init__(self, name = ""):
        self.name = name
        self.data_dic = {}
        self.index = -1

    class User():
        def __init__(self, id, firstName, lastName, password, email, mobile, address, authority):
            self.id = id
            self.firstName = firstName
            self.lastName = lastName
            self.password = password
            self.email = email
            self.mobile = mobile
            self.address = address
            self.authority = authority
    #return struct item.
    def make_struct(self, id, firstName, lastName, password, email, mobile, address, authority):
        return self.User(id, firstName, lastName, password, email, mobile, address, authority)

    class Train():
        def __init__(self, id, speed, coordinate, shortestDistance, state, fire, condition, direction, line, nextStation):
            self.id = id
            self.speed = speed
            self.coordinate = coordinate
            self.shortestDistance = shortestDistance
            self.state = state
            self.fire = fire
            self.condition = condition
            self.direction = direction
            self.line = line
            self.nextStation = nextStation
    def make_struct(self, id, speed, coordinate, shortestDistance, state, fire, condition, direction, line, nextStation):
        return self.Train(id, speed, coordinate, shortestDistance, state, fire, condition, direction, line, nextStation)

    class Control():
        def __init__(self, time, weather, railway, detectorCondition, speedLimit, redSignal, timeDifference):
            self.time = time
            self.weather = weather
            self.railway = railway
            self.detectorCondition = detectorCondition
            self.speedLimit = speedLimit
            self.redSignal = redSignal
            self.timeDifference = timeDifference
    def make_struct(self, time, weather, railway, detectorCondition, speedLimit, redSignal, timeDifference):
        return self.Control(time, weather, railway, detectorCondition, speedLimit, redSignal, timeDifference)

    class Line():
        def __init__(self, id, coordinate, direction, schedule):
            self.id = id
            self.coordinate = coordinate
            self.direction = direction
            self.schedule = schedule
    def make_struct(self, id, coordinate, direction, schedule):
        return self.Line(id, coordinate, direction, schedule)

    class Station():
        def __init__(self, id, coordinate, platform):
            self.id = id
            self.coordinate = coordinate
            self.platform = platform
    def make_struct(self, id, coordinate, direction, schedule):
        return self.Station(id, coordinate, direction, schedule)

    class Schedule():
        def __init__(self, id, line, optimalArriveTime):
            self.id = id
            self.line = line
            self.optimalArriveTime = optimalArriveTime
    def make_struct(self, id, line, optimalArriveTime):
        return self.Schedule(id, line, optimalArriveTime)