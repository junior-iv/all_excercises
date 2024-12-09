from typing import Optional


class Node:
    father: Optional['Node']
    children: list
    name: str
    distance_to_father: float

    def __init__(self, name: Optional[str]):
        self.father = None
        self.children = []
        self.name = name
        self.distance_to_father = 0

    def __str__(self) -> str:
        return self.get_name(self, True)

    @classmethod
    def subtree_to_newick(cls, node: Optional['Node'], reverse: bool = False) -> str:
        """This method is for internal use only."""
        node_list = node.children[::-1] if reverse else node.children
        if node_list:
            result = '('
            for child in node_list:
                result += (f'{cls.subtree_to_newick(child, reverse) if child.children else child.name}:'
                           f'{child.distance_to_father},')
            result = result[:-1] + ')'
        else:
            result = f'{node.name}:{node.distance_to_father}'
        return result

    @classmethod
    def get_name(cls, node: Optional['Node'], is_full_name: bool = False) -> str:
        return (f'{cls.subtree_to_newick(node) if node.children and is_full_name else node.name}:'
                f'{node.distance_to_father:.5f}')

    def add_child(self, child: Optional['Node'], distance_to_father: float) -> None:
        self.children.append(child)
        child.father = self
        child.distance_to_father = distance_to_father

    def get_full_distance_to_leafs(self) -> float:
        list_result = []
        child = self
        while True:
            list_result.append({'node': child, 'distance': child.distance_to_father})
            if not child.children:
                break
            child = child.children[0]
        return sum([i['distance'] for i in list_result])

    def get_full_distance_to_father(self) -> float:
        list_result = []
        father = self
        while True:
            list_result.append({'node': father, 'distance': father.distance_to_father})
            if not father.father:
                break
            father = father.father
        return sum([i['distance'] for i in list_result])
