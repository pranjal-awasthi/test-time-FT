from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import networkx as nx
from networkx.algorithms import tree

# generic class for describing a solution to an optimization problem
class Solution:
    def __init__(self, **kwargs):
        pass

    # return the objective function values associated with the solution
    def evaluate(self) -> Dict[str, float]:
        pass

    # get the LLM response string from the solution object
    def get_response_string_from_solution(self) -> str:
        pass

# generic class for describing a data point for an optimization problem
class DataPoint:
    def __init__(self, **kwargs):
        pass

    # generate the data point instance given the kwargs
    def generate(self):
        pass

    # convert data point to prompt for LLM call
    def convert_to_prompt(self) -> str:
        pass

    # parse LLM output to a solution object
    def parse_to_solution(self, solution: str) -> Solution:
        pass

    def generate_random_feasible_solution(self) -> str:
        pass


    # generate finetuning data from a sample of responses
    def generate_ft_data(self, samples: List[str]) -> List[Dict[str, str]]:
        data = []
        prompt = self.convert_to_prompt()
        for sample in samples:
            inner_dict = {}
            solution = self.parse_to_solution(sample)
            response = solution.get_response_string_from_solution()
            inner_dict["query"] = prompt
            inner_dict["response"] = response
            data.append(inner_dict)

        return data


class CustomDataset(torch.utils.data.Dataset):

  def __init__(self, tokenizer: PreTrainedTokenizer, ft_dataset: List[Dict[str, str]], max_length: int = 768):

    super().__init__()

    self.max_length = max_length
    self.n = len(ft_dataset)

    self.input_ids = []
    self.attn_masks = []

    for data_point in ft_dataset:

      encodings_dict = tokenizer('<|startoftext|>'+ data_point['query'] + "\n" + data_point['response'] + '<|endoftext|>',
                                 truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))


  def __len__(self):
    return self.n

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]



### ------------------ Class defs for the clustering problem ------------------ ###
COLORS = ["red", "blue", "green"]

class ClusteringSolution(Solution):

    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.is_valid = kwargs.get("is_valid", False)
        self.total_cost = kwargs.get("total_cost", np.inf)
        self.total_color_violation = kwargs.get("total_color_violation", np.inf)
        self.clusters = kwargs.get("clusters", [])

    def get_response_string_from_solution(self) -> str:
        # self.clusters is a list of length 3 (one per color). Each c in self.clusters is a list of clusters, i.e, a list of lists
        response = ""
        for index in range(len(COLORS)):
          c = self.clusters[index]
          response += f"""{colors[index]} clusters: """
          for ell in c:
            for i in ell:
              response += f"""{i} """
          response = response.strip() + "\n"
        return response

    def evaluate(self) -> Dict[str, float]:
        return {"total_cost": self.total_cost, "total_color_violation": self.total_color_violation}


class ClusteringDataPoint(DataPoint):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.num_points = kwargs.get("num_points", 24)
        self.r = kwargs.get("r", 10)
        self.data_points = []
        self.dist = []
        self.dist_dict = []
        self.colors = []


    def generate(self):
        self.data_points = [int(i) for i in np.arange(self.num_points)]
        np.random.shuffle(self.data_points)

        col_index = 0
        # assign distance 1 to points in the optimal clustering
        for i in range(0,self.num_points,2):
            self.dist.append((self.data_points[i], self.data_points[i+1], 1))
            self.dist_dict[(self.data_points[i], self.data_points[i+1])] = 1
            self.dist_dict[(self.data_points[i+1], self.data_points[i])] = 1
            self.colors[self.data_points[i]] = self.cols[col_index]
            self.colors[self.data_points[i+1]] = self.cols[col_index]
            col_index = (col_index + 1) % 3

        # pick a few random pairs of same color and assing a distance of r
        for _ in range(self.num_points//2):
            i = int(np.random.randint(0,self.num_points))
            j = int(np.random.randint(0,self.num_points))
            if i != j and ((i,j,1) not in self.dist) and ((j,i,1) not in self.dist) and (self.colors[i] == self.colors[j]):
                self.dist.append((i,j,self.r))
                self.dist_dict[(i,j)] = self.r
                self.dist_dict[(j,i)] = self.r

        # pick a few random pairs of different colors and assign a distance of 1
        for _ in range(self.num_points//2):
            i = int(np.random.randint(0,self.num_points))
            j = int(np.random.randint(0,self.num_points))
            if self.colors[i] != self.colors[j] and ((i,j,1) not in self.dist) and ((j,i,1) not in self.dist) and (((i,j,self.r) not in self.dist)) and (((j,i,self.r) not in self.dist)):
                self.dist.append((i,j,1))
                self.dist_dict[(i,j)] = 1
                self.dist_dict[(j,i)] = 1


        np.random.shuffle(self.dist)


    def convert_to_prompt(self) -> str:
        prompt = f"""You are given {self.num_points} data points numbered from 0 to {self.num_points-1}.
        """
        prompt += f""" The distance (denoted as d(x,y)) between some of the pairs of points is given as follows:\n"""
        for (i,j,r) in self.dist:
            prompt += f"""Distance between {i} and {j} is: {r} \n """


        prompt += f""" Each point also has a color as follows: \n"""
        for i in range(self.num_points):
            prompt += f""" Color of point {i} is: {self.colors[i]} \n """

        for color in ["red", "blue", "green"]:
          prompt += f""" Hence in total the point with color {color} are: """
          for i in range(self.num_points):
            if self.colors[i] == color:
              prompt += f"""{i}, """

          prompt = prompt.strip(" ").strip(",")
          prompt += "\n"

        prompt += f""" For any pair x,y for which the distance is not explicitly provided, assume that the distance is {self.r} if x and y have the same color. Otherwise assume that
        the distance between x and y is 1.\n"""


        prompt +=  f""" Your goal is to divide the given points into {self.num_points//2} non-empty and disjoint clusters. \n
    """

        prompt += f""" You must ensure that each cluster only has points of the same color. \n"""

        prompt += f""" You MUST ensure that each cluster is non-empty. \n"""

        prompt += f""" At the same time you must minimize the k-median cost of the clustering which equals the sum of the 1-median cost of the clusters.
        For a given cluster its 1-median cost is defined as the sum of the distances of all the points in the cluster to a center point (chosen among them so as to minimize the cost). \n
        """

        prompt += f""" Do not provide code or pseudocode. If you cannot solve the problem give me your best guess."""

        prompt += f""" Format your final answer strictly in the following manner: 'red clusters:' followed by a list of points that belong to red colored clusters (space separated, two per cluster)
        \n 'blue clusters:' followed by a list of points that belong to blue colored clusters (space separated, two per cluster) \n
           'green clusters:' followed by a list of points that belong to green colored clusters (space separated, two per cluster) \n
        """


        return prompt


    def parse_to_solution(self, response: str) -> Solution:
        clusters = []
        n = self.num_points

        for color in COLORS:
          c = []
          col_str = response.split(f"{color} clusters:")[2].split("\n")[0].strip()
          ell = col_str.split(" ")
          index = 0
          try:
            while index < len(ell):
              if index + 1 < len(ell):
                c.append([int(ell[index].strip()), int(ell[index+1].strip())])
                index += 2
              else:
                c.append([int(ell[index].strip())])
                index += 1
          except:
              return ClusteringSolution({"is_valid": False})

          clusters.append(c)

        # compute total cost and total color violation
        total_cost = 0
        total_color_violation = 0
        is_covered = {}

        total_clusters = 0
        for index in range(3):
          total_clusters += len(clusters[index])

        # check if total clusters equals the desired amount
        if total_clusters != self.num_points//2:
          print(f"total clusters violated: {total_clusters}")
          return ClusteringSolution({"is_valid": False})

        for index in range(3):
          for c in clusters[index]:
            for i in c:
              if i in is_covered:
                is_covered[i] += 1
              else:
                is_covered[i] = 1

        for index in range(3):
          for c in clusters[index]:
              for i in range(len(c)):
                  for j in range(len(c)):
                      if c[i] not in self.colors or c[j] not in self.colors:
                          return ClusteringSolution({"is_valid": False})
                      if self.colors[c[i]] != self.colors[c[j]]:
                          total_color_violation += 1

              best_k_median_cost = np.inf
              for i in range(len(c)):
                  inner_cost = 0
                  for j in range(len(c)):
                      if i == j:
                          continue
                      if (c[i],c[j]) in self.dist_dict:
                          inner_cost += self.dist_dict[(c[i],c[j])]
                      elif (c[j],c[i]) in self.dist_dict:
                          inner_cost += self.dist_dict[(c[j],c[i])]
                      elif self.colors[c[i]] != self.colors[c[j]]:
                          inner_cost += 1
                      else:
                          inner_cost += self.r

                  best_k_median_cost = min(best_k_median_cost, inner_cost)

              total_cost += best_k_median_cost

        for i in range(self.num_points):
            if i not in is_covered:
                print(f"{i} not covered\n")
                return ClusteringSolution({"is_valid": False})
            elif is_covered[i] > 1:
                print(f"{i} covered more than once\n")
                return ClusteringSolution({"is_valid": False})

        return ClusteringSolution({"is_valid": True, "total_cost": total_cost, "total_color_violation": total_color_violation//2, "clusters": clusters})

    # generate a random solution that preserves the coloring constraint -- this is used for initial instruction tuning of the base model
    def generate_random_ft_data_point(self) -> Dict[str, str]:
      data = {}
      data["query"] = self.convert_to_prompt()
      r = []
      b = []
      g = []
      for i in range(self.num_points):
        if self.colors[i] == 'red':
          r.append(i)
        elif self.colors[i] == 'blue':
          b.append(i)
        else:
          g.append(i)

      np.random.shuffle(r)
      np.random.shuffle(b)
      np.random.shuffle(g)

      colors = []
      for c in r:
        colors.append('red')
      for c in b:
        colors.append('blue')
      for c in g:
        colors.append('green')

      indices = r+b+g

      response = ""
      response += "red clusters: "
      for i in r[:-1]:
        response += f"""{i} """
      response += f"""{r[-1]}\n"""

      response += "blue clusters: "
      for i in b[:-1]:
        response += f"""{i} """
      response += f"""{b[-1]}\n"""

      response += "green clusters: "
      for i in g[:-1]:
        response += f"""{i} """
      response += f"""{g[-1]}\n"""
      data["response"] = response

      return data




### ------------------ Class defs for the minimum spanning tree (MST) problem ------------------ ###

class MSTSolution(Solution):

    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.is_valid = kwargs.get("is_valid", False)
        self.total_cost = kwargs.get("total_cost", np.inf)
        self.total_deg_violation = kwargs.get("total_deg_violation", np.inf)
        self.edges = kwargs.get("edges", [])

    def get_response_string_from_solution(self) -> str:
        response = ""
        for a,b in self.edges:
            response += f"({a}, {b})\n"
        response = response.strip("\n")
        return respnonse

    def evaluate(self) -> Dict[str, float]:
        return {"total_cost": self.total_cost, "total_deg_violation": self.total_deg_violation}


class MSTDataPoint(DataPoint):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.num_lines = kwargs.get("num_lines", 4)
        self.prob = kwargs.get("prob", 0.2)
        self.deg = kwargs.get("deg", 2)
        self.edges = []


    def generate(self):
        n = self.num_lines * self.num_lines
        even = 0
        startx = starty = 0
        endx = endy = num_lines-1
        gg = []

        for i1 in range(startx, endx+1):
            for j1 in range(starty, endy+1):
                for i2 in range(startx, endx+1):
                    for j2 in range(starty, endy+1):
                        if (i1 != i2 or j1 != j2) and np.random.uniform(0,1) < self.prob:
                            gg.append((i1 * num_lines + j1,i2 * num_lines + j2))

        for j in range(starty, endy+1):
            for i in range(startx, endx):
                gg.append((i * num_lines + j,(i+1) * num_lines + j))

        for j in range(starty, endy):
            if even == 0:
                gg.append((endx * num_lines + j,endx * num_lines + j+1))
            else:
                gg.append((startx * num_lines + j,startx * num_lines + j+1))
            even = 1 - even

        self.edges = list(set(gg))

    def convert_to_prompt(self) -> str:
        prompt = f"""You are given a graph with vertices labeled from 0 to {self.num_lines * self.num_lines-1}. Each line below lists an edge of the graph as (i,j).\n"""

        for e in self.edges:
            prompt += f"{e}\n"

        prompt += f"""Your goal is to output a list of H of a subset of the edges that form a spanning tree, i.e., the subgraph induced by H should be connected. Furthermore, each vertex should
        appear in at most {self.deg} times in the list. Simply output the list of edges and nothing else. Format your answer by producing one edge per new line."""

        return prompt

    # generates a random spanning tree
    def generate_random_mst(self) -> List[Tuple[int, int]]:
        mst = []
        visited = {}
        num_visited = 0

        # pick random row
        i = np.random.randint(0,self.num_lines * self.num_lines)

        num_visited += 1
        visited[i] = -1
        curr_vertex = i

        while num_visited < self.num_lines**2:

            # get neighbors of current vertex
            i = curr_vertex
            neighbors = []
            for e in self.edges:
                p, r = e
                if (p == i):
                    neighbors.append(r)
                elif (r == i):
                    neighbors.append(p)

            index = np.random.randint(0,len(neighbors))

            k = neighbors[index]

            if k not in visited:
                visited[k] = i
                num_visited += 1
                mst.append((i,k))

            curr_vertex = k

        return mst

    # convert an mst to forest to preserve the nodewise degree constraints
    def convert_mst_to_forest(self, mst: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        forest = []
        degrees = {}
        for v in mst:
            i, k = v
            if i in degrees and degrees[i] == self.deg:
                continue
            elif k in degrees and degrees[k] == self.deg:
                continue
            if i not in degrees:
                degrees[i] = 1
            else:
                degrees[i] += 1

            if k not in degrees:
                 degrees[k] = 1
            else:
                degrees[k] += 1

            forest.append(v)

        return forest


    # generate a random solution that preserves the degree constraint -- this is used for initial instruction tuning of the base model
    def generate_random_ft_data_point(self) -> Dict[str, str]:
        mst = self.generate_random_mst()
        forest = self.convert_mst_to_forest(mst)
        data = {}
        data["query"] = self.convert_to_prompt()
        data["response"] = ""
        for a,b in forest:
            if (a,b) not in self.edges and (b,a) in self.edges:
                data["response"] += f"({b}, {a})\n"
            elif (a,b) in self.edges:
                data["response"] += f"({a}, {b})\n"
        data["response"] = data["response"].strip("\n")

        return data

    def get_mst_degree_profile(self, mst: List[Tuple[int, int]]) -> List[int]:
        degrees = {}
        for v in mst:
            i, k = v
        if i not in degrees:
            degrees[i] = 1
        else:
            degrees[i] += 1
        if k not in degrees:
            degrees[k] = 1
        else:
            degrees[k] += 1

        deg_vals = degrees.values()
        return list(deg_vals)

    def get_num_connected_components(self, forest: List[Tuple[int, int]]) -> int:

        G = nx.Graph()
        for i in range(self.num_lines):
            for j in range(self.num_lines):
                G.add_node(i * self.num_lines + j)
        G.add_edges_from(forest)

        return nx.number_connected_components(G)


    def reduce_to_forest(self, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        G = nx.Graph()
        for i in range(self.num_lines):
            for j in range(self.num_lines):
                G.add_node(i * self.num_lines + j)
        G.add_edges_from(edges)
        forest = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        return list(forest)

    def cleanup_sample(self, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        cleaned_edges = []
        for e in edges:
            a,b = e
            if (a,b) in self.edges or (b,a) in self.edges:
                cleaned_edges.append(e)
        if len(s2) == 0:
            cleaned_edges.append(self.edges[0])

        return cleaned_edges


    def parse_to_solution(self, response: str) -> Solution:
        output_str = response.split("one edge per new line.\n")[-1].split("<|endoftext|>")[0]
        edges = []
        output_str = output_str.strip("\n").strip(" ")
        for e in output_str.split("\n"):
          if e == "":
            s.append(self.edges[0])
          else:
            a = e.strip("(").strip(")")
            L = a.split(",")
            i = int(L[0].strip(" "))
            l = int(L[1].strip(" "))
            edges.append((i,l))

        cleaned_edges = self.cleanup_sample(edges)
        forest = self.reduce_to_forest(cleaned_edges)
        deg_profile = self.get_mst_degree_profile(forest)
        num_violation = np.sum(np.array(deg_profile) > self.deg)
        num_cc = self.get_num_connected_components(forest)
        return MSTSolution({"is_valid": True, "edges": forest, "total_cost": num_cc-1, "total_deg_violation": num_violation})


### ------------------ Class defs for the minimum spanning tree (MST) problem ------------------ ###

class LineSchedulingSolution(Solution):

    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.is_valid = kwargs.get("is_valid", False)
        self.total_cost = kwargs.get("total_cost", np.inf)
        self.total_box_violation = kwargs.get("total_box_violation", np.inf)
        self.visit_times = kwargs.get("visit_times", [])

    def get_response_string_from_solution(self) -> str:
        response = ""
        for t in self.visit_times:
            response += f"{t}\n"
        response = response.strip("\n")
        return response

    def evaluate(self) -> Dict[str, float]:
        return {"total_cost": self.total_cost, "total_box_violation": self.total_box_violation}

class LineSchedulingDataPoint(DataPoint):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.num_points = kwargs.get("num_points", 7)
        self.travel_time_range = kwargs.get("travel_time_range", [1,20])
        self.box_constraint_range = kwargs.get("box_constraint_range", [1,20])
        self.box_constraints = []
        self.travel_times = []
        self.opening_times = []

    def generate(self):

        for j in range(self.num_points):
            visit_durations = np.random.randint(self.box_constraint_range[0], self.box_constraint_range[1], size=2)
            self.box_constraints.append(np.sort(visit_durations))


        self.travel_times = np.random.randint(self.travel_time_range[0], self.travel_time_range[1], size=n-1)
        self.opening_times = [0]

        # pick opening time such that you need to visit each location for the maximum amount to avoid waiting
        for j in range(1,self.num_points):
            self.opening_times.append(self.opening_times[j-1] + self.box_constraints[j-1][1] + self.travel_times[j-1])

    # given visit durations compute the total wait time
    def compute_total_wait_time(self, visit_durations: List[float]) -> float:
        wait_time = 0.0
        curr_time = 0.0

        if(len(visit_durations) < self.num_points):
            visit_durations += [0]*(self.num_points-len(visit_durations))

        for i in range(self.num_points):
            wait_time += max(0, self.opening_times[i] - curr_time)
            curr_time += max(0, self.opening_times[i] - curr_time) + visit_durations[i]
            if i < self.num_points-1:
                curr_time += self.travel_times[i]
        return wait_time

    # given visit durations compute total box costraint violation
    def compute_box_violation(data_point, visit_durations):
        box_violation = 0.0

        if(len(visit_durations) < self.num_points):
            visit_durations += [0]*(self.num_points-len(visit_durations))

        for i in range(self.num_points):
            if visit_durations[i] > self.box_constraints[i][1]:
                box_violation += visit_durations[i] - self.box_constraints[i][1]
            elif visit_durations[i] < self.box_constraints[i][0]:
                box_violation +=  self.box_constraints[i][0] - visit_durations[i]

        return box_violation

    def convert_to_prompt(self) -> str:
        n = self.num_points
        m = n-1

        prompt = f"""
            There are {n} museums on a line. They are numbered from 0 to {m}.
            I'm currently located at museum 0 and the current timestamp is 0.
            I want to visit all these museums one by one in a sequence. Each museum has an opening time.
            If I reach a particular museum before it opens then I may have to wait. The opening times for the museums are as follows:[
            """
        for i in self.opening_times:
            prompt += f"{int(i)}, "

        prompt += "]. "

        prompt += f"""In addition, the following list of {m} numbers contains the time to travel from museum i to i+1.
        So the first number is the time to travel from 0 to 1 and so on: ["""

        for i in self.travel_times:
            prompt += f"{int(i)}, "

        prompt += "]. "


        prompt += """Finally, I have certain constraints in terms of the minimum and maximum amount of time I want to visit each museum.
        This is described as the following list of arrays: ["""

        for i in self.box_constraints:
            prompt += f"""[{int(i[0])}, {int(i[1])}], """
        prompt += "]. "

        prompt += f"""Give me a schedule in terms of a list of {n} numbers
        describing how much time I should spend at each place so that all my constraints are satisfied and at the same time
        my total wait time is as little as possible. Do not use code. Simply output the list of {n} numbers (one per line) and nothing else."""

        return prompt        


    def parse_to_solution(self, response: str) -> Solution:
        output_str = response.split("nothing else.\n")[-1].split("<|endoftext|>")[0]
        # Truncate values larger to satisy box constraints upper range
        visit_times = []
        for e in output_str.split("\n"):
            try:
                float_val = int(e)
            except ValueError:
                float_val = 0
            visit_times.append(min(float_val,self.box_constraint_range[1]))

        visit_times = visit_times[:self.num_points]
        box_violation = self.compute_box_violation(visit_times)
        total_cost = self.compute_total_wait_time(visit_times)

        return LineSchedulingSolution({"is_valid": True, "total_cost": total_cost, "total_box_violation": total_box_violation, "visit_times": visit_times})

