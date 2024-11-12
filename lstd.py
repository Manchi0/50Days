class Node:
    def __init__(self,value,parent=None):
        self.val=value
        self.check=0
        self.parent=parent
        self.children=[]

    
    def add_children(self,*children):
        for child in children:
            if type(child)==Node:
                child.parent=self
                self.children.append(child)
            else:
                print(f'{child} is not a node, please intialize it first' )


queue=[]



# def bfs_assistant(queue):
#     node=queue[0]
#     queue.pop(0)
#     if node.check==0:
#         queue+=node.children
#         node.check=1
#     return queue

# def bfs(source,queue):
#     queue=[source]
#     while len(queue)!=0:
#         queue=bfs_assistant(queue)




# def bfs2(source,queue):
#     traversal=[]
#     queue=[source]
#     while len(queue)>0:
#         node=queue[0]
#         traversal.append(queue.pop(0))
#         if node.check==0:
#             queue+=node.children
#             node.check=1


def bfs3(source,queue):
    traversal=[]
    queue=[source]
    while len(queue)>0:
        node=queue[0]
        traversal.append(node.val)
        for child in node.children:
            if child.check==0:
                queue.append(child)
                child.check=1
    return traversal
        
    


if __name__ == "__main__":
    # Create nodes
    root = Node(1)
    child1 = Node(2)
    child2 = Node(3)
    child3 = Node(4)
    child4 = Node(5)
    child5 = Node(6)
    child6 = Node(7)

    # Build the tree structure
    root.add_children(child1, child2)
    child1.add_children(child3, child4)
    child2.add_children(child5, child6)

    # Expected BFS traversal: 1, 2, 3, 4, 5, 6, 7
    result = bfs3(root)
    print("BFS Traversal Result:", result)

    # Reset check values for nodes if needed for repeated testing
    for node in [root, child1, child2, child3, child4, child5, child6]:
        node.check = 0

# class edge:
#     def __init__(self,capacity,p_node,c_node):
        
# x=Node(1)
# ch=Node(2)
# x.add_children(ch)
# print(ch.parent.val)
