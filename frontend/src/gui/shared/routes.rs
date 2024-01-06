use std::rc::Rc;

const PATH_HOME: &'static str = "home";
const PATH_ASSISTANT: &'static str = "assistant";
const PATH_TEST: &'static str = "test";
const PATH_TEST_TEST1: &'static str = "test1";
const PATH_TEST_TEST2: &'static str = "test2";

pub struct RouteBuilder {
    route: Rc<BaseRoute>,
}

impl RouteBuilder {
    pub fn new(base_href: String) -> RouteBuilder {
        let mut base_href = base_href;
        if !base_href.is_empty() && base_href.chars().last().unwrap() == '/' {
            base_href.pop();
        }
        let route = Rc::new(BaseRoute {
            parent: None,
            path: base_href,
        });
        return RouteBuilder { route };
    }

    pub fn home(&self) -> BaseRoute {
        BaseRoute {
            parent: Some(self.route.clone()),
            path: PATH_HOME.to_string(),
        }
    }

    pub fn assistant(&self) -> BaseRoute {
        BaseRoute {
            parent: Some(self.route.clone()),
            path: PATH_ASSISTANT.to_string(),
        }
    }

    pub fn test(&self) -> TestRoute {
        let route = Rc::new(BaseRoute {
            parent: Some(self.route.clone()),
            path: PATH_TEST.to_string(),
        });
        TestRoute { route }
    }
}

pub struct TestRoute {
    route: Rc<BaseRoute>,
}

impl TestRoute {
    pub fn test1(&self) -> BaseRoute {
        BaseRoute {
            parent: Some(self.route.clone()),
            path: PATH_TEST_TEST1.to_string(),
        }
    }

    pub fn test2(&self) -> BaseRoute {
        BaseRoute {
            parent: Some(self.route.clone()),
            path: PATH_TEST_TEST2.to_string(),
        }
    }

    pub fn full_path(&self) -> String {
        return self.route.full_path();
    }
}

pub struct BaseRoute {
    parent: Option<Rc<BaseRoute>>,
    path: String,
}

impl BaseRoute {
    fn build_full_path_reversed(&self, mut reversed_paths: Vec<String>) -> Vec<String> {
        reversed_paths.push(self.path.clone());

        match self.parent.clone() {
            Some(existing_parent) => existing_parent.build_full_path_reversed(reversed_paths),
            None => reversed_paths,
        }
    }

    pub fn full_path(&self) -> String {
        let mut paths = self.build_full_path_reversed(vec![]);
        paths.reverse();
        return paths.join("/");
    }
}

#[cfg(test)]
mod tests {

    use wasm_bindgen_test::wasm_bindgen_test;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[wasm_bindgen_test]
    fn test_home() {
        let home_path = RouteBuilder::new("/app/".to_string()).home().full_path();
        assert_eq!(home_path, "/app/home".to_string());
    }

    #[wasm_bindgen_test]
    fn test_home_no_basehref() {
        let home_path = RouteBuilder::new("".to_string()).home().full_path();
        assert_eq!(home_path, "/home".to_string());
    }

    #[wasm_bindgen_test]
    fn test_assistant() {
        let home_path = RouteBuilder::new("/app/".to_string())
            .assistant()
            .full_path();
        assert_eq!(home_path, "/app/assistant".to_string());
    }

    #[wasm_bindgen_test]
    fn test_assistant_no_basehref() {
        let home_path = RouteBuilder::new("".to_string()).assistant().full_path();
        assert_eq!(home_path, "/assistant".to_string());
    }

    #[wasm_bindgen_test]
    fn test_test() {
        let home_path = RouteBuilder::new("/app/".to_string()).test().full_path();
        assert_eq!(home_path, "/app/test".to_string());
    }

    #[wasm_bindgen_test]
    fn test_test_no_basehref() {
        let home_path = RouteBuilder::new("".to_string()).test().full_path();
        assert_eq!(home_path, "/test".to_string());
    }

    #[wasm_bindgen_test]
    fn test_test1() {
        let home_path = RouteBuilder::new("/app/".to_string())
            .test()
            .test1()
            .full_path();
        assert_eq!(home_path, "/app/test/test1".to_string());
    }

    #[wasm_bindgen_test]
    fn test_test1_no_basehref() {
        let home_path = RouteBuilder::new("".to_string()).test().test1().full_path();
        assert_eq!(home_path, "/test/test1".to_string());
    }

    #[wasm_bindgen_test]
    fn test_test2() {
        let home_path = RouteBuilder::new("/app/".to_string())
            .test()
            .test2()
            .full_path();
        assert_eq!(home_path, "/app/test/test2".to_string());
    }

    #[wasm_bindgen_test]
    fn test_test2_no_basehref() {
        let home_path = RouteBuilder::new("".to_string()).test().test2().full_path();
        assert_eq!(home_path, "/test/test2".to_string());
    }
}
