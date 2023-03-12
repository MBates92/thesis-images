import numpy as np
import matplotlib.pyplot as plt


class perimeter_area():
    def __init__(self, field, granularity = 20):
        granularity = granularity + 2
        field = np.pad(field,10)
        self.field_min, self.field_max = np.min(field), np.max(field)
        levels = np.linspace(self.field_min,self.field_max, granularity)
        self.levels = levels[1:-1]
        contours = plt.contour(field, self.levels)
        plt.close()
        self.areas = []
        self.perimeters = []
        for i in range(len(self.levels)):
            contour = contours.collections[i]
            vs = contour.get_paths()
            area=0
            perimeter = 0
            for island in vs:
                x = island.vertices[:,0]
                y = island.vertices[:,1]
                pi = np.sum([np.sqrt(x**2+y**2) for x,y in zip(x,y)])
                ai = 0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
                area += np.abs(ai)
                perimeter += pi
            self.areas.append(area)
            self.perimeters.append(perimeter)

        self.gradient, self.intercept = np.polyfit(np.log10(self.areas), np.log10(self.perimeters), deg=1)
        self.fractal_dimension = self.gradient*2
        self.beta = 6-2*self.fractal_dimension
        self.hurst = (self.beta-2)/2