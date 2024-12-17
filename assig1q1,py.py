import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def show(name, n, m,i, Title):
     plt.subplot( n,m,i)
     plt.imshow(name, cmap='gray')
     plt.title(Title)
     plt.axis('off')

img = mpimg.imread('MrUCL.webp')
plt.figure(figsize=(10, 5))

show(img, 1, 2, 1, "First View")
show(img, 1, 2, 2, "Second View")

plt.tight_layout()
plt.show()

plt.savefig('task1_output.png')
