class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def display_info(self):
        print(f"Car: {self.year} {self.make} {self.model}")

    
class ElectricCar(Car):
    def __init__(self, make, model, year, battery_size):
        super().__init__(make, model, year)
        self.battery_size = battery_size
        self.__battery_life = 100

    def display_info(self):
        super().display_info()
        print(f"Battery Size: {self.battery_size} kWh")

    def __calculate_battery_life(self):
        # 私有方法模拟电池使用情况
        return self.__battery_life - 10

    def get_battery_life(self):
        updated_life = self.__calculate_battery_life()
        print(f"Current battery life: {updated_life}%")


my_car = Car('Toyota', 'Corolla', 2021)
my_car.display_info()

my_tesla = ElectricCar('Tesla', 'Model X', 2019, 100)
my_tesla.display_info()
my_tesla.get_battery_life()