import volscore_wrapper as vw

def test_simple_vol():
    vs = vw.VolScore()
    mock_data = [100, 101, 105, 102, 98, 99]
    rv = vs.computeRealizedVol(mock_data)
    vol_score = vs.computeVolScore(mock_data)
    print("Realized Vol:", rv)
    print("VolScore:", vol_score)

if __name__ == "__main__":
    test_simple_vol()
