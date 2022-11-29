import bendstiff
import pickle
import numpy as np

### Integration test 
def test_full_integration():
    par_file = open("test/test_par.pkl", "rb")
    par = pickle.load(par_file)
    par_file.close()
    
    result_file = open("test/test_results.pkl", "rb")
    result = pickle.load(result_file)
    result_file.close()
    par['plot'] = False
    new_result =  bendstiff.run.run_bendstiff('test/test_image.jpg', par)
    
    tol= 1e-8

    assert np.allclose(new_result[0],result[0], atol=tol) # x-coordinate
    assert np.allclose(new_result[1],result[1], atol=tol) # y-coordinate
    assert np.allclose(new_result[2],result[2], atol=tol) # curvature
    assert np.allclose(new_result[3],result[3], atol=tol) # moment
    

### Unit tests 

def test_line_kernel():
    test_kernel = np.array([[0,1], [1,0], [1,0]])
    assert (test_kernel == bendstiff.image_processing.line_kernel(2, np.pi/3)).all()
  
    

def test_b_matrixl():
    
    x = [0, 1, 2, 3]
    p = 2
    t = bendstiff.curvefit.knots(x, 2)
    
    test_B = np.array([[1,0,0,0,0], [0,0.5,0.5,0,0], [0,0,0.5,0.5,0], [0,0,0,0,1]])
    
    assert (test_B == bendstiff.curvefit.b_matrix(np.array(x), t, p)).all()
    

# Make a test of line kernel 
