from node import Node
from typing import Optional, List, Union, Dict


class Tree:
    root: Optional[Node]

    def __init__(self, data: Union[str, Node, None] = None) -> None:
        if isinstance(data, str):
            self.newick_to_tree(data)
        elif isinstance(data, Node):
            self.root = data
        else:
            self.root = Node('root')

    def __str__(self) -> str:
        return self.get_newick()

    def __eq__(self, other):
        return self.get_newick() == other.get_newick()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.get_node_count() < other.get_node_count()

    def __le__(self, other):
        newick1 = self.get_newick()
        newick2 = other.get_newick()
        return self.get_node_count() < other.get_node_count() or newick1 == newick2 or len(newick1) < len(newick2)

    def __gt__(self, other):
        return self.get_node_count() > other.get_node_count()

    def __ge__(self, other):
        newick1 = self.get_newick()
        newick2 = other.get_newick()
        return self.get_node_count() > other.get_node_count() or newick1 == newick2 or len(newick1) > len(newick2)

    def print_node_list(self, reverse: bool = False, with_additional_details: bool = False, mode: Optional[str] = None,
                        filters: Optional[Dict[str, list[Union[float, int, str, list[float]]]]] = None) -> None:
        """
        Print a list of nodes.

        This function prints a list of nodes. If the `reverse` argument is set to `True`, the list
        of nodes will be printed in reverse order. By default, `reverse` is `False`, so the list
        will be printed in its natural order.

        Args:
            reverse (bool, optional): If `True`, print the nodes in reverse order. If `False` (default),
                                      print the nodes in their natural order.
            with_additional_details (bool, optional): `False` (default)
            mode (Optional[str]): `None` (default), 'pre-order', 'in-order', 'post-order'.
            filters (Dict, optional):
        Returns:
            None: This function does not return any value; it only prints the nodes to the standard output.
        """
        data_structure = self.list_nodes_info(self.root, reverse, with_additional_details, mode, filters)

        str_result = ''
        for i in data_structure:
            str_result = f'{str_result}\n{i}'
        print(str_result, '\n')

    def get_list_nodes_info(self, reverse: bool = False, with_additional_details: bool = False, mode: Optional[str] =
                            None, filters: Optional[Dict[str, list[Union[float, int, str, list[float]]]]] = None
                            ) -> List[Dict[str, Union[float, bool, str, list[float]]]]:
        """
        Args:
            reverse (bool, optional): If `True`, the resulting list of nodes will be in reverse order.
                                      If `False` (default), the nodes will be listed in their natural
                                      traversal order.
            with_additional_details (bool, optional): `False` (default).
            mode (Optional[str]): `None` (default), 'pre-order', 'in-order', 'post-order'.
            filters (Dict, optional):
        """
        return self.list_nodes_info(self.root, reverse, with_additional_details, mode, filters)

    def get_node_count(self, filters: Optional[Dict[str, list[Union[float, int, str, list[float]]]]] = None) -> int:
        """
        Args:
            filters (Dict, optional):
        """
        return len(self.get_list_nodes_info(False, True, None, filters))

    @staticmethod
    def add_nodes_info(permit: int, list_result: List[Dict[str, Union[float, bool, str]]], info: Dict[str,
                       Union[float, bool, str, list[float]]]):
        if permit:
            list_result.append(info)

    @staticmethod
    def list_nodes_info(node: Node, reverse: bool = False, with_additional_details: bool = False, mode: Optional[str] =
                        None, filters: Optional[Dict[str, list[Union[float, int, str, list[float]]]]] = None
                        ) -> List[Dict[str, Union[float, bool, str, list[float]]]]:
        """
        Retrieve a list of descendant nodes from a given node, including the node itself or retrieve a list of
        descendant nodes from the current instance of the `Tree` class.

        This function collects all child nodes of the specified `node`, including the `node` itself, or collects all
        child nodes of the current instance of the `Tree` class if `node` is not provided. The function
        returns these nodes names as a list.

        Args:
            node (Node, optional): The node whose child nodes, along with itself, are to be collected. If not
                            specified, collects all child nodes of the current instance are collected.
            reverse (bool, optional): If `True`, the resulting list of nodes will be in reverse order.
                                      If `False` (default), the nodes will be listed in their natural
                                      traversal order.
            with_additional_details (bool, optional): `False` (default).
            mode (Optional[str]): `None` (default), 'pre-order', 'in-order', 'post-order'.
            filters (Dict, optional):
        Returns:
            list: A list of nodes names including the specified `node` (or the current instance's nodes  names) and its
                                    children. The list is ordered according to the `reverse` argument.
        """
        list_result = []
        mode = 'pre-order' if mode is None or mode.lower() not in ('pre-order', 'in-order', 'post-order') else (
            mode.lower())

        def get_list(newick_node: Node) -> None:
            nonlocal list_result
            lavel = 1
            full_distance = [newick_node.distance_to_father]
            father = newick_node.father
            if father:
                father_name = father.name
                node_type = 'node'
                while father:
                    full_distance.append(father.distance_to_father)
                    lavel += 1
                    father = father.father
            else:
                father_name = ''
                node_type = 'root'

            if not newick_node.children:
                node_type = 'leaf'

            info = {'node': newick_node.name, 'distance': full_distance[0],
                    'lavel': lavel, 'node_type': node_type, 'father_name': father_name,
                    'full_distance': full_distance}
            if filters:
                permit = 0
                for key in filters.keys():
                    for value in filters[key]:
                        permit += sum(k == key and info[k] == value for k in info)
            else:
                permit = 1

            if permit and mode == 'pre-order':
                list_result.append(info if with_additional_details else newick_node.name)

            for i, child in enumerate(newick_node.children[::-1]) if reverse else enumerate(newick_node.children):
                get_list(child)
                if permit and mode == 'in-order' and i == 0:
                    list_result.append(info if with_additional_details else newick_node.name)
            if not newick_node.children:
                if permit and mode == 'in-order':
                    list_result.append(info if with_additional_details else newick_node.name)

            if permit and mode == 'post-order':
                list_result.append(info if with_additional_details else newick_node.name)

        get_list(node)
        return list_result

    def get_newick(self, reverse: bool = False) -> str:
        """
        Convert the current tree structure to a Newick formatted string.

        This function serializes the tree into a Newick format, which is a standard format for representing
        tree structures. If the `reverse` argument is set to `True`, the order of the tree nodes in the
        resulting Newick string will be reversed. By default, `reverse` is `False`, meaning the nodes
        will appear in their natural order.

        Args:
            reverse (bool, optional): If `True`, reverse the order of the nodes in the Newick string.
                                      If `False` (default), preserve the natural order of the nodes.

        Returns:
            str: A Newick formatted string representing the tree structure.
        """
        return f'{Node.subtree_to_newick(self.root, reverse)};'

    def find_node_by_name(self, name: str, node: Optional[Node] = None) -> bool:
        """
        Search for a node by its name in a tree structure.

        This function searches for a node with a specific name within a tree. If a root node is provided,
        the search starts from that node; otherwise, it searches from the default root of the tree.
        The function returns `True` if a node with the specified name is found, and `False` otherwise.

        Args:
            name (str): The name of the node to search for. This should be the exact name of the node
                        as a string.
            node (Node, optional): The node from which to start the search. If not provided,
                                        the search will start from the default root node of the tree.

        Returns:
            bool: `True` if a node with the specified name is found; `False` otherwise.
        """
        node = self.root if node is None else node
        if name == node.name:
            return True
        else:
            for child in node.children:
                if self.find_node_by_name(name, child):
                    return True
        return False

    def newick_to_tree(self, newick: str) -> Optional['Tree']:
        """
        Convert a Newick formatted string into a tree object.

        This function parses a Newick string, which represents a tree structure in a compact format,
        and constructs a tree object from it. The Newick format is often used in phylogenetics to
        describe evolutionary relationships among species.

        Args:
            newick (str): A string in Newick format representing the tree structure. The string
                              should be properly formatted according to Newick syntax.

        Returns:
            Tree: An object representing the tree structure parsed from the Newick string. The tree
                  object provides methods and properties to access and manipulate the tree structure.
        """
        newick = newick.replace(' ', '').strip()
        if newick.startswith('(') and newick.endswith(';'):

            len_newick = len(newick)
            list_end = [i for i in range(len_newick) if newick[i:i + 1] == ')']
            list_start = [i for i in range(len_newick) if newick[i:i + 1] == '(']
            list_children = []

            num = self.__counter()

            while list_start:
                int_start = list_start.pop(-1)
                int_end = min([i for i in list_end if i > int_start]) + 1
                list_end.pop(list_end.index(int_end - 1))
                node_name = newick[int_end: min([x for x in [newick.find(':', int_end), newick.find(',', int_end),
                                                 newick.find(';', int_end), newick.find(')', int_end)] if x >= 0])]
                distance_to_father = newick[int_end + len(node_name): min([x for x in [newick.find(',', int_end),
                                                                          newick.find(';', int_end), newick.find(')',
                                                                          int_end)] if x >= 0])]

                (visibility, node_name) = (True, node_name) if node_name else (False, 'nd' + str(num()).rjust(4, '0'))

                sub_str = newick[int_start:int_end]
                list_children.append({'children': sub_str, 'node': node_name, 'distance_to_father': distance_to_father,
                                      'visibility': visibility})

            list_children.sort(key=lambda x: len(x['children']), reverse=True)

            for i in range(len(list_children)):
                for j in range(i + 1, len(list_children)):
                    node_name = list_children[j]['node'] if list_children[j]['visibility'] else ''
                    list_children[i]['children'] = list_children[i]['children'].replace(
                        list_children[j]['children'] + node_name, list_children[j]['node'])
            for dict_children in list_children:
                if list_children.index(dict_children):
                    node = self.__find_node_by_name(dict_children['node'])
                    node = node if node else self.__set_node(
                        dict_children['node'] + dict_children['distance_to_father'], num)
                else:
                    node = self.__set_node(dict_children['node'] + dict_children['distance_to_father'], num)
                    self.root = node

                self.__set_children_list_from_string(dict_children['children'], node, num)
            return self

    def get_html_tree(self, style: str = '', status: str = '') -> str:
        """This method is for internal use only."""
        return self.structure_to_html_tree(self.tree_to_structure(), style, status)

    def tree_to_structure(self, reverse: bool = False) -> dict:
        """This method is for internal use only."""
        return self.subtree_to_structure(self.root, reverse)

    def add_distance_to_father(self, distance_to_father: float = 0) -> None:
        def add_distance(node: Node) -> None:
            nonlocal distance_to_father
            node.distance_to_father += distance_to_father
            node.distance_to_father = round(node.distance_to_father, 12)
            for child in node.children:
                add_distance(child)

        add_distance(self.root)

    def get_edges_list(self, reverse: bool = False) -> List[str]:
        list_result = []

        def get_list(node: Node) -> None:
            nonlocal list_result
            if node.father:
                list_result.append((node.father.name, node.name))
            for child in node.children[::-1] if reverse else node.children:
                get_list(child)

        get_list(self.root)
        return list_result

    @classmethod
    def __get_html_tree(cls, structure: dict, status: str) -> str:
        """This method is for internal use only."""
        tags = (f'<details {status}>', '</details>', '<summary>', '</summary>') if structure['children'] else ('', '',
                                                                                                               '', '')
        str_html = (f'<li> {tags[0]}{tags[2]}{structure["name"].strip()} \t ({structure["distance_to_father"]}) '
                    f'{tags[3]}')
        for child in structure['children']:
            str_html += f'<ul>{cls.__get_html_tree(child, status)}</ul>\n' if child[
                'children'] else f'{cls.__get_html_tree(child, status)}'
        str_html += f'{tags[1]}</li>'
        return str_html

    @classmethod
    def get_robinson_foulds_distance(cls, tree1: Union['Tree', str], tree2: Union['Tree', str]) -> float:
        """This method is for internal use only."""
        tree1 = Tree(tree1) if type(tree1) is str else tree1
        tree2 = Tree(tree2) if type(tree2) is str else tree2

        edges_list1 = sorted(Tree.get_edges_list(tree1), key=lambda item: item[1])
        edges_list2 = sorted(Tree.get_edges_list(tree2), key=lambda item: item[1])

        distance = 0
        for node in edges_list1:
            distance += 0 if edges_list2.count(node) else 1
        for node in edges_list2:
            distance += 0 if edges_list1.count(node) else 1

        return distance / 2

    @classmethod
    def structure_to_html_tree(cls, structure: dict, styleclass: str = '', status: str = '') -> str:
        """This method is for internal use only."""
        return (f'<ul {f" class = {chr(34)}{styleclass}{chr(34)}" if styleclass else ""}>'
                f'{cls.__get_html_tree(structure, status)}</ul>')

    @classmethod
    def subtree_to_structure(cls, node: Node, reverse: bool = False) -> dict:
        """This method is for internal use only."""
        dict_node = {'name': node.name.strip(), 'distance_to_father': node.distance_to_father}
        list_children = []
        if node.children:
            for child in node.children[::-1] if reverse else node.children:
                list_children.append(cls.subtree_to_structure(child, reverse))
        dict_node.update({'children': list_children})
        return dict_node

    def __find_node_by_name(self, name: str, node: Optional[Node] = None) -> Optional[Node]:
        """This method is for internal use only."""
        node = self.root if node is None else node
        if name == node.name:
            return node
        else:
            for child in node.children:
                node = self.__find_node_by_name(name, child)
                if node:
                    return node
        return None

    def __set_children_list_from_string(self, str_children: str, father: Optional[Node], num) -> None:  # List[Node]:
        """This method is for internal use only."""
        str_children = str_children[1:-1] if str_children.startswith('(') and str_children.endswith(
            ')') else str_children
        lst_nodes = str_children.split(',')
        for str_node in lst_nodes:
            node = self.__set_node(str_node.strip(), num)
            node.father = father
            father.children.append(node)

    @staticmethod
    def check_newick(newick_text: str) -> bool:
        return newick_text.startswith('(') and newick_text[:-1].endswith(')') and newick_text.endswith(';')

    @staticmethod
    def __set_node(node_str: str, num) -> Node:
        """This method is for internal use only."""
        if node_str.find(':') > -1:
            node_data = node_str.split(':')
            node_data[0] = node_data[0] if node_data[0] else 'nd' + str(num()).rjust(4, '0')
            try:
                node_data[1] = float(node_data[1])
            except ValueError:
                node_data[1] = 0.0
        else:
            node_data = [node_str if node_str else 'nd' + str(num()).rjust(4, '0'), 0.0]

        node = Node(node_data[0])
        node.distance_to_father = float(node_data[1])
        return node

    @staticmethod
    def __counter():
        """This method is for internal use only."""
        count = 0

        def sub_function():
            nonlocal count
            count += 1
            return count

        return sub_function
