if __name__ == '__main__':
    import json

    with open('result.json') as f:
        res1 = json.loads(f.read())
    with open('result_itt=False.json') as f:
        res2 = json.loads(f.read())
        
    def merge(d1, d2):
        def dfs(t1, t2):
            if not isinstance(t1, dict):
                return {
                    'teacher trainable': t1,
                    'teacher fixed': t2
                }
                
            res = dict()
            for k, v in t1.items():
                res[k] = dfs(v, t2[k])
            
            return res
        
        return dfs(d1, d2)
        
    res = merge(res1, res2)
    
    with open('merged.json', 'w') as f:
        f.write(json.dumps(res))