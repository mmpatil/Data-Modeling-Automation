'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('IndependentVariableResult', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      ModelId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'ModelRunDetail',
          key: 'Id'
        }
      },
      RunId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'RunDetail',
          key: 'Id'
        }
      },
      Name: Sequelize.STRING,
      Coefficient: Sequelize.FLOAT,
      Pval: Sequelize.FLOAT,
      Transformations: Sequelize.STRING,
      VIF: Sequelize.STRING
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('IndependentVariableResult');
  }
};
