#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)

# bar for apples
a = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[0], width=.5, color='r',
            label='apples')

# bar for bananas
b = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[1], width=.5,
            color='yellow', label='bananas', bottom=fruit[0])

# bar for oranges
o = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[2], width=.5,
            color='#ff8000', label='oranges', bottom=fruit[0] + fruit[1])

# bar for peaches
p = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[3], width=.5,
            color='#ffe5b4', label='peaches',
            bottom=fruit[0] + fruit[1] + fruit[2])

plt.yticks(np.arange(0, 81, 10))
plt.legend()
plt.show()
