ba: ba.cpp
	g++ --std=c++11 ba.cpp dataio.cpp util.cpp -I /usr/include/eigen3/ -lpoplar -lpopops -lpoplin -lpoputil -ltbb -lboost_program_options -o ba

slam: slam.cpp
	g++ --std=c++11 slam.cpp dataio.cpp util.cpp -I /usr/include/eigen3/ -lpoplar -lpopops -lpoplin -lpoputil -ltbb -lboost_program_options -o slam

clean:
	rm -f ba
	rm -f slam
