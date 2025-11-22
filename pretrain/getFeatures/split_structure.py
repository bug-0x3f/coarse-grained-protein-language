import copy
from typing import Dict, List, Any, Optional


class DsspStructureSplitter:
    
    ALL_TYPES = ['H', 'B', 'S', 'T', 'G', 'I', 'E']
    
    SPECIAL_TYPES = ['S', 'T', 'B']
    
    def __init__(self):
        """Initialize the DsspStructureSplitter."""
        pass
    
    @staticmethod
    def _init_blank() -> Dict[str, List]:
        return {
            'list': [],
            'seq': []
        }
    
    def split_dssp_data(self, dsspfile: str) -> Dict[int, Dict[str, Any]]:
        """Split DSSP file data into structural fragments.
        
        Args:
            dsspfile: Path to the DSSP file.
            
        Returns:
            Dictionary where keys are fragment IDs and values contain:
                - 'list': List of residue IDs
                - 'seq': List of amino acid sequences
                - 'structure': Secondary structure type
        """
        with open(dsspfile, 'r') as f:
            dssp_text = f.readlines()
        
        result = {}  
        wid = 0
        process_flag = False 
        last_structure = '#'  
        cnt_blank = 0
        blanklist = copy.deepcopy(self._init_blank())
        cnt_unk = 0
        sumlines = 0
        cnt_line = -1
        
        for line in dssp_text:
            cnt_line += 1
            tmp = line.strip()
            if tmp.startswith('#'):
                process_flag = True
                continue
            
            if process_flag:
                residue_id = line[7:11].strip()  
                structure = line[16]
                AA = line[13]
                
                if residue_id == '' or AA == 'X':
                    cnt_unk += 1
                    continue
                
                sumlines += 1
                if structure in self.ALL_TYPES:
                    if cnt_blank != 0 and cnt_blank < 3:
                        if (wid in result.keys() and result[wid]['structure'] in self.SPECIAL_TYPES):
                            result[wid]['list'] += blanklist['list']
                            last_structure = result[wid]['structure']
                            result[wid]['seq'] += blanklist['seq']
                            blanklist = copy.deepcopy(self._init_blank())
                            cnt_blank = 0
                        elif structure not in self.SPECIAL_TYPES:
                            wid += 1
                            result[wid] = {}
                            result[wid]['list'] = blanklist['list']
                            result[wid]['structure'] = 'C'  
                            result[wid]['seq'] = blanklist['seq']
                            blanklist = copy.deepcopy(self._init_blank())
                    
                    elif cnt_blank > 0:  
                        wid += 1
                        result[wid] = {}
                        result[wid]['list'] = blanklist['list']
                        result[wid]['structure'] = 'C' 
                        result[wid]['seq'] = blanklist['seq']
                        blanklist = copy.deepcopy(self._init_blank())
                        cnt_blank = 0
                    
                    if structure == last_structure:
                        result[wid]['list'].append(residue_id)
                        result[wid]['seq'].append(AA)
                    else:  
                        wid += 1
                        
                        if cnt_blank != 0 and cnt_blank < 3: 
                            result[wid] = {}
                            result[wid]['list'] = blanklist['list']
                            result[wid]['seq'] = blanklist['seq']
                        if wid not in result.keys():
                            result[wid] = {}
                            result[wid]['list'] = [residue_id]
                            result[wid]['seq'] = [AA]
                        else:
                            result[wid]['list'].append(residue_id)
                            result[wid]['seq'].append(AA)
                        result[wid]['structure'] = structure
                    blanklist = copy.deepcopy(self._init_blank())
                    cnt_blank = 0
                elif structure == ' ':
                    cnt_blank += 1
                    blanklist['list'].append(residue_id)
                    blanklist['seq'].append(AA)
                else:
                    print('Unknown structure:', structure)
                    pass
                last_structure = structure
        
        if cnt_blank != 0 and cnt_blank < 3 and wid in result.keys() and result[wid]['structure'] in self.SPECIAL_TYPES:
            result[wid]['list'] += blanklist['list']
            result[wid]['seq'] += blanklist['seq']
            last_structure = result[wid]['structure']
            blanklist = copy.deepcopy(self._init_blank())
            cnt_blank = 0
        else:  
            if cnt_blank > 0:
                wid += 1
                result[wid] = {}
                result[wid]['list'] = blanklist['list']
                result[wid]['seq'] = blanklist['seq']
                result[wid]['structure'] = 'C'  
                blanklist = copy.deepcopy(self._init_blank())
                cnt_blank = 0
        
        return result


_default_splitter = DsspStructureSplitter()


def split_dssp_data(dsspfile: str) -> Dict[int, Dict[str, Any]]:
    return _default_splitter.split_dssp_data(dsspfile)

def filter_frag(res: Dict[int, Dict[str, Any]], min_length: int = 3, max_length: int = 60) -> List[int]:
    ret = []
    for key, value in res.items():
        seq = value['seq']
        length = len(seq)
        if length < min_length or length > max_length:
            ret.append(key)
    return ret



