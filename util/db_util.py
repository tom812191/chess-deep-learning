import psycopg2


def connect(settings):
    return psycopg2.connect("dbname='{db_name}' user='{user}' host='{host}' password='{password}' port='{port}'"
                            .format(**settings))


def create_table(conn, name, cols, recreate=False):
    cur = conn.cursor()

    # Drop the table if it already exists
    cur.execute(
        "select exists(select * from information_schema.tables where table_name='{0}')".format(name))
    if cur.fetchone()[0]:
        if not recreate:
            return

        cur.execute("drop table if exists {0}".format(name))

    # Create the table
    query = """
                CREATE TABLE IF NOT EXISTS {0} ({1});
            """.format(name, ', '.join(cols))
    cur.execute(query)
    conn.commit()