
class Distance(object):
    def __init__(self, number, unit):
        self.to_cm_multiplier = {
            'um': 10 ** -4,
            'mm': 0.1,
            'cm': 1,
            'm': 100,
            'in': 2.54,
            'ft': 30.48
        }
        self.distance_in_cm = self.determine_distance_in_cm(number, unit)

    def determine_distance_in_cm(self, number, unit):
        return self.to_cm_multiplier[unit] * number

    def represented_as(self, unit):
        return (1.0 / self.to_cm_multiplier[unit]) * self.distance_in_cm

    def add(self, other):
        distance_in_cm = self.distance_in_cm + other.distance_in_cm
        return Distance(distance_in_cm, 'cm')

    def subtract(self, other):
        distance_in_cm = self.distance_in_cm - other.distance_in_cm
        return Distance(distance_in_cm, 'cm')


class Mass(object):
    def __init__(self, number, unit):
        self.to_g_multiplier = {
            'mg': 0.001,
            'g': 1,
            'kg': 1000,
            'lb': 454.592,
            'tonnes': 10 ** 6
        }
        self.mass_in_g = self.determine_mass_in_g(number, unit)

    def determine_mass_in_g(self, number, unit):
        return self.to_g_multiplier[unit] * number

    def add(self, other):
        mass_in_g = self.mass_in_g + other.mass_in_g
        return Mass(mass_in_g, 'g')

    def subtract(self, other):
        mass_in_g = self.mass_in_g - other.mass_in_g
        return Mass(mass_in_g, 'g')


def main():
    x = Distance(1, 'cm')
    print('1 cm in mm: {}'.format(x.represented_as('mm')))
    print('1 cm in ft: {}'.format(x.represented_as('ft')))

    y = Distance(3, 'in')
    print('3 in to um: {}'.format(y.represented_as('um')))
    print('3 in to in: {}'.format(y.represented_as('in')))
    print('3 in to ft: {}'.format())

    z = x.add(y)
    print(z.distance_in_cm)


if __name__ == '__main__':
    main()




