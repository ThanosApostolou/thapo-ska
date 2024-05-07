//
// ATTEMPT 3
//
export type Result<O, E> = Ok<O, E> | Err<O, E>;

export class Ok<O, E> {
    private readonly _data: O;

    private constructor(data: O) {
        this._data = data;
    }

    get data() {
        return this._data;
    }

    isOk(): this is Ok<O, E> {
        return true;
    }

    isErr<E>(): this is Err<O, E> {
        return false;
    }

    unwrap(): O {
        return this._data;
    }

    /**
     * Maps a Result<O, E> to Result<U, E> by applying a function to a contained Ok value, leaving an Err value untouched
     */
    map<U>(mapFunction: (data: O) => U): Result<U, E> {
        return new Ok(mapFunction(this._data));
    }

    /**
     * Maps a Result<O, E> to Result<O, I> by applying a function to a contained Err value, leaving an Ok value untouched.
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    mapErr<I>(_mapFunction: (error: E) => I): Result<O, I> {
        return new Ok(this._data);
    }

    /**
     * Creates a new Result<O, E> with ok data
     */
    static new<O, E>(data: O): Result<O, E> {
        return new Ok(data);
    }

}

export class Err<O, E> {
    private readonly _error: E;

    private constructor(error: E) {
        this._error = error;
    }

    get error() {
        return this._error;
    }

    isOk<O>(): this is Ok<O,E> {
        return false;
    }

    isErr(): this is Err<O, E> {
        return true;
    }

    unwrap(): O {
        throw this._error;
    }

    /**
     * Maps a Result<O, E> to Result<U, E> by applying a function to a contained Ok value, leaving an Err value untouched
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    map<U>(_mapFunction: (data: O) => U): Result<U, E> {
        return new Err(this._error);
    }

    /**
     * Maps a Result<O, E> to Result<O, I> by applying a function to a contained Err value, leaving an Ok value untouched.
     */
    mapErr<I>(mapFunction: (error: E) => I): Result<O, I> {
        return new Err(mapFunction(this._error));
    }

    /**
     * Creates a new Err
     */
    static new<O, E>(error: E): Err<O, E> {
        return new Err(error);
    }

}
