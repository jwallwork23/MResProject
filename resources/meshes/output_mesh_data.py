# Open mesh file
mesh = open("tohoku_edit.msh", "rw+")
print "Name of the file: ", mesh.name

for line in mesh:
    print line

mesh.close()
