COMPOSE = docker compose -f ./docker-compose.yml

all:	
	${COMPOSE} up -d --build

clean:
	rm -rf ./db/*
	rm -rf ./chroma_db/*

fclean: clean
	${COMPOSE} down -v
	rm -rf ./backend/db
	rm -rf ./backend/chroma_db

down:
	${COMPOSE} down -v

delimages:
	docker rmi -f $$(docker images -aq)

re: down all

logs:
	${COMPOSE} logs -f

exec-backend:
	${COMPOSE} exec -it backend sh

exec-frontend:
	${COMPOSE} exec -it frontend sh

.PHONY: all datadir clean fclean down delimages re